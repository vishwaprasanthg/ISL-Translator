import torch
from ml.models.static_model import StaticHandCNN

def test_static_model_output_shape():
    """Verify that the model accepts (B, 126) and outputs (B, num_classes)"""
    num_classes = 26
    model = StaticHandCNN(num_classes=num_classes)
    
    # Create dummy input of batch size 4, 126 features
    dummy_input = torch.randn(4, 126)
    
    # Forward pass
    output = model(dummy_input)
    
    assert output.shape == (4, num_classes), f"Expected shape (4, {num_classes}), got {output.shape}"
    assert output.dtype == torch.float32

def test_static_model_gradients():
    """Verify that gradients propagate back to input"""
    num_classes = 10
    model = StaticHandCNN(num_classes=num_classes)
    
    dummy_input = torch.randn(2, 126, requires_grad=True)
    output = model(dummy_input)
    
    # Simulate a loss
    loss = output.sum()
    loss.backward()
    
    # Input should have gradients
    assert dummy_input.grad is not None
    assert dummy_input.grad.shape == (2, 126)
