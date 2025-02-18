from PIL import Image
import torch
import torchvision.transforms as T

t = torch.jit.load("test.pt")
tensor_list = list(t.parameters())

param_names = [name for name in model.state_dict()]
model_dict = model.state_dict()

for tensor, param_name in zip(tensor_list, param_names):
    model_dict[param_name] = tensor
    
model.load_state_dict(model_dict)
# Load and preprocess the image
image_path = "/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/test_images/BRATS_004_z93.png"
image = Image.open(image_path).convert("L")  # Convert to grayscale (if necessary)

# Define the necessary transformations (e.g., resizing, tensor conversion)
transform = T.Compose([
     # Resize to the required input size (adjust as needed)
    T.ToTensor(),           # Convert image to a tensor
     # Normalize (adjust mean/std as needed)
])

# Apply transformations
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension (unsqueeze)

# Run the image through the model
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_tensor)  # Get model output

# Postprocess the output (e.g., convert to a mask)
output = output.squeeze(0)  # Remove batch dimension
output = output.argmax(0)   # Get the class with the highest score for each pixel

# Convert output to a PIL image (for visualization)
output_image = Image.fromarray(output.cpu().numpy().astype('uint8'))

# Save or display the output
output_image.save("output_mask.png")
output_image.show()
