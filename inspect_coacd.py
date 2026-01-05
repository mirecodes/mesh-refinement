import coacd
import inspect

print("Signature:")
try:
    print(inspect.signature(coacd.run_coacd))
except Exception as e:
    print(e)

print("\nDocstring:")
print(coacd.run_coacd.__doc__)
