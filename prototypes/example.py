import jax
import jax.numpy as jnp
import y337


def f(devices, a, b):
	a = devices[0](lambda x: x + 1)(a)
	b = devices[1](lambda x: x * 2)(b)
	c = devices[0].recv(b)
	d = devices[0](jnp.add)(a, c)
	e = devices[1].recv(d)
	return c, e

def g(devices, a):
	c = devices[0](lambda x: x + x)(a)
	b = devices[1].recv(c)
	d = devices[1](lambda x: x + x)(b)
	e = devices[0].recv(d)
	return b, e


def main(mesh):
	print("Results", mesh.remote_execute(g)(jnp.ones((2, 2))))

if __name__ == '__main__':
	y337.godmode(main)

