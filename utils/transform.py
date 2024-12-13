from torchvision import transforms

class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target):
        # Apply the image transformations
        img = self.transform(img)

        # Apply transformations to the target (such as resizing bounding boxes)
        # In this case, let's assume you're just returning the target as-is, but you could apply operations like resizing boxes.
        return img, target
