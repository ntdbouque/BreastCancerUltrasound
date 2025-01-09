import tenseal as ts 
import torch  
import torchvision.transforms as T  
from model import ConvNet, train, test  
from model_encrypt import EncConvNet, enc_test, infer  
from data import load_dataloader 


## CONFIG:
DATASET_PATH = '/workspace/competitions/Sly/CV_Final_Final/data/train'
TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
])
INPUT_SIZE = (28,28)
TEST_SIZE=0.2

CHECKPOINT_PATH = '/workspace/competitions/Sly/CV_Final_Final/model/checkpoint_encrypt_clf.pt'

# Run:
## Training Old Model:
# train_loader, test_loader = load_dataloader(
#     dataset_path=DATASET_PATH,
#     test_size=TEST_SIZE,
#     input_size=INPUT_SIZE,
#     transform=TRANSFORM_TRAIN
# )

# model = ConvNet()
#criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model = train(model, train_loader, criterion, optimizer, 50)

# torch.save(model.state_dict(), CHECKPOINT_PATH)

model = ConvNet()
model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
model.eval()
#test(model, test_loader, criterion)

## Encryption Parameters
bits_scale = 26

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

context.global_scale = pow(2, bits_scale)

context.generate_galois_keys()

# ## Reset some model param:
# _, test_loader = load_dataloader(
#     dataset_path=DATASET_PATH,
#     test_size=TEST_SIZE,
#     input_size=INPUT_SIZE,
#     transform=TRANSFORM_TRAIN,
#     batch_size=1
# )
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]


CUSTOM_TRANSFORM = T.Compose([
    T.Resize((28, 28)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
])

from PIL import Image
image = Image.open('/workspace/competitions/Sly/CV_Final_Final/data/train/normal/normal (7)_mask.png')
image = CUSTOM_TRANSFORM(image).squeeze(0)
print(image.shape)
# Testing Env model:
enc_model = EncConvNet(model)
#enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
output = infer(context, enc_model, image, kernel_shape, stride)
print("Label: ",output)