import tenseal as ts 
import torch  
import torchvision.transforms as T  
from model import ConvNet, train, test  
from model_encrypt import EncConvNet, enc_test  
from data import load_dataloader 


## CONFIG:
DATASET_PATH = '/workspace/competitions/Sly/CV_Final_Final/data/train'
TRANSFORM_TRAIN = T.Compose([
    T.ToTensor(),
])
INPUT_SIZE = (28,28)
TEST_SIZE=0.2

# Run:
## Training Old Model:
train_loader, test_loader = load_dataloader(
    dataset_path=DATASET_PATH,
    test_size=TEST_SIZE,
    input_size=INPUT_SIZE,
    transform=TRANSFORM_TRAIN
)

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, train_loader, criterion, optimizer, 50)

test(model, test_loader, criterion)

## Encryption Parameters
bits_scale = 26

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

context.global_scale = pow(2, bits_scale)

context.generate_galois_keys()

## Reset some model param:
_, test_loader = load_dataloader(
    dataset_path=DATASET_PATH,
    test_size=TEST_SIZE,
    input_size=INPUT_SIZE,
    transform=TRANSFORM_TRAIN,
    batch_size=1
)
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

# Testing Env model:
enc_model = EncConvNet(model)
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
