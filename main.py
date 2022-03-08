import cv2
from PIL import Image
import matplotlib.pyplot as plt


def to_pixels(path):
    original = Image.open(path)
    print('original.size: ', original.size)

    small_image1 = original.resize((48, 48), Image.BOX)
    print('small size 1: ', small_image1.size)
    result_image1 = small_image1.resize(original.size, Image.NEAREST)

    plt.imshow(result_image1)
    plt.title(' 48x 48 ')
    plt.show()

    path = path.split('_')[1]
    result_image1.save('final_'+path)


def detect_face(path):
    img = cv2.imread(path)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector.detectMultiScale(gray, 1.3, 5)
    try:
        for (x, y, w, h) in face:
            if len(list(filter(lambda t: t >= 50, face[0]))) == 4:
                roi_color = img[y-50:y + h + 50, x - 50:x + w+50]
            else:
                roi_color = img[y:y + h, x:x + w]
        resized = cv2.resize(roi_color, (128, 128))
        n_path = 'f_'+path
        cv2.imwrite('f_'+path, resized)
        return n_path
    except:
        print("no faces detected")
    return ""


# This program has 4 images of people.
# The User chose 1, and gets a face-only, pixelized picture.
# I'm using plt as showing the saved image.
# Prefixes:
#   f_*image.jpeg* : cropped image. has only the face.
#   final_*image.jpeg : pixelized face.
if __name__ == '__main__':
    inn = input('chose number:\n')
    path = 'f' + inn + '.jpeg'
    path = detect_face(path)
    to_pixels(path)
