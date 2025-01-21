import jax

@jax.jit
def arbitrary_function(x):
    return x * 2
  
print(arbitrary_function(3))  

x = jax.grad(arbitrary_function) 

print(x(3.0))