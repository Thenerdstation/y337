from typing import NamedTuple, Any, Callable
import jax
import jax.numpy as jnp
import cloudpickle
from functools import reduce


def make_jaxpr_picklable(j):
	j.jaxpr.eqns = [jax.core.new_jaxpr_eqn(x.invars, x.outvars, x.primitive, x.params) for x in j.jaxpr.eqns]

class PlacedMethod(NamedTuple):
	func: Any
	placement: Any
	in_vars: Any
	out_vars: Any

	def execute(self, placement, env):
		if self.placement is placement:
			results = self.func(*[env[v] for v in self.in_vars])
			out_vars, _ = jax.tree_flatten(results)
			for v, res in zip(self.out_vars, out_vars):
				assert v not in env, f"Overwriting {v} from {env} should never happen"
				env[v] = res
	
	def map_new_placements(self, placement_map, var_map):
		return PlacedMethod(
			func=self.func,
			placement=placement_map[self.placement],
			in_vars=[var_map[v] for v in self.in_vars],
			out_vars=[var_map[v] for v in self.out_vars])


class MessagePass(NamedTuple):
	in_var: Any
	out_var: Any

	def execute(self, placement, env):
		from mpi4py import MPI
		import mpi4jax
		comm = MPI.COMM_WORLD
		if self.out_var.placement is self.in_var.placement and self.out_var.placement is placement:
			env[self.out_var] = env[self.in_var] # No-op
			return
		if self.out_var.placement is placement:
			env[self.out_var], _ = mpi4jax.recv(
				jnp.zeros(self.out_var.shaped_array.shape, dtype=self.out_var.shaped_array.dtype),
				source=self.in_var.placement.rank, comm=comm)
		if self.in_var.placement is placement:
			mpi4jax.send(env[self.in_var], dest=self.out_var.placement.rank, comm=comm)

	def map_new_placements(self, placement_map, var_map):
		return MessagePass(
			in_var=var_map[self.in_var],
			out_var=var_map[self.out_var])


class PlacedShapedArray(NamedTuple):
	shaped_array: Any
	placement: Any

	def __hash__(self):
		return id(self)

	def __eq__(self, other):
		return id(self) == id(other)

class Mesh:
	def __init__(self, client, num_devices):
		self.client = client
		self.num_devices = num_devices


	def remote_execute(self, func):
		ops = []
		devices = [Placement(rank=i, builder_callback=ops.append) for i in range(2, self.num_devices + 2)]

		def wrapper(*args):
			placed_args = [PlacedShapedArray(shaped_array=val, placement=None) for val in args]
			outvars, _ = jax.tree_flatten(func(devices, *placed_args), is_leaf=lambda x: isinstance(x, PlacedShapedArray))
			remote_results = []
			for device in devices:
				def execution(*args):
					env = {parg: arg for parg, arg in zip(placed_args, args)}
					for op in ops:
						op.execute(device, env)
					return {i: env[x] for i, x in enumerate(outvars) if x.placement is device}
				values = self.client.submit(jax.jit(execution), *args, workers=device.rank)
				remote_results.append(values)
			results = reduce(lambda a, b: {**a, **b}, [x.result() for x in remote_results])
			final_results = [results[i] for i in range(len(results))]
			return final_results
		return wrapper


class Placement(NamedTuple):
	rank: Any
	builder_callback: Callable

	def __call__(self, f):
		def wrapper(*args):
			assert all(map(lambda x: x.placement is self or x.placement is None, args)), "Some input not in this placement."
			result = jax.eval_shape(f, *[x.shaped_array for x in args])
			out_arrays, tree = jax.tree_flatten(result)
			out_arrays = [PlacedShapedArray(shaped_array=x, placement=self) for x in out_arrays]
			self.builder_callback(PlacedMethod(func=f, placement=self, in_vars=args, out_vars=out_arrays))
			return jax.tree_unflatten(tree, out_arrays)
		return wrapper

	def recv(self, val):
		outvar = PlacedShapedArray(shaped_array=val.shaped_array, placement=self)
		self.builder_callback(MessagePass(
			in_var=val, 
			out_var=outvar))
		return outvar


def godmode(main):
	from dask_mpi import initialize
	initialize()
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	from dask.distributed import Client
	import dask
	with dask.config.set({'scheduler.work-stealing': False}):
		client = Client()
		num_workers = comm.Get_size() - 2
		assert num_workers > 0, "You must use atleast 3 processes."
		main(Mesh(client=client, num_devices=num_workers))


