import pennylane.numpy as np
import pennylane as qml

def test_lighting_gpu():
    """Test that the lighting GPU device works correctly."""
    dev = qml.device("lightning.gpu", wires=2)
    print("Device initialized:", dev)
    
test_lighting_gpu()
print("Test passed successfully.")