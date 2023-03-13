import argparse
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
# Define transformations for input images
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize(256),  # resize the input image to 256x256
    transforms.CenterCrop(224),  # crop the center 224x224 region of the image
    transforms.ToTensor(),  # convert the image to a tensor with values between 0 and 1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         # normalize the image using the mean and standard deviation for ImageNet
                         std=[0.229, 0.224, 0.225])
])


# Define the function to classify a given image
def classify_image(image_path, model, classes):
    # Load the image and apply the transformations
    image = Image.open(image_path)
    image_tensor = transform(image).cuda()
    image_tensor = image_tensor.unsqueeze(0)  # add a batch dimension to the tensor
    # Use the model to predict the class probabilities for the input image
    with torch.no_grad():
        predictions = model(image_tensor)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        # Print the top 5 predicted class labels and their corresponding probabilities
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        return top5_idx.tolist(), top5_prob.tolist()


def main(args):
    # Load the pre-trained model
    model = torch.hub.load('pytorch/vision:v0.6.0', args.model, pretrained=True)
    model.eval()  # set the model to evaluation mode
    model.cuda()
    # Define the class labels for ImageNet
    with open('imagenet_classes.txt') as f:
        classes = [line.strip().split(' ')[1] for line in f.readlines()]

    # List all images files in the directory
    image_paths = glob.glob(os.path.join(args.dataset_path, "*/*/*.jpg"), recursive=True)
    image_paths.sort()

    with open(args.output_path, 'w') as f:
        for image_path in tqdm(image_paths):
            top5_idx, top5_prob = classify_image(image_path, model, classes)
            result = f"{os.path.basename(image_path)}"
            for idx, prob in zip(top5_idx, top5_prob):
                result += f",{classes[idx]} {prob}"
            print(result, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r'D:\Data\afhq-old')
    parser.add_argument('--model', type=str, default='resnet152', choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'])
    parser.add_argument('--output_path', type=str, default='labels.csv')
    cfg = parser.parse_args()
    main(cfg)
