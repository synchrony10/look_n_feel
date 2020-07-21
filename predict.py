import sys
import cv2
from keras.models import load_model
from matplotlib import pyplot as plt
import time

model = load_model("models/model.h5")

def find_faces(image):
    face_cascade = cv2.CascadeClassifier('models/detection/haarcascade_frontalface_default.xml')
    face_rects = face_cascade.detectMultiScale(
        image,
        scaleFactor = 1.01,
        minNeighbors = 22
    )
    return face_rects


def load_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray_image


def predict(gray_image, rects):

    # face_rects = find_faces(gray_image)
    # for face_rect in face_rects:
    x, y, w, h = rects
    face = gray_image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48)).reshape((1, 48, 48, 1))
    predicted_emotions = model.predict(face)[0]
    best_emotion = 'happiness' if predicted_emotions[1] > predicted_emotions[0] else 'neutral'

    # Create a json serializable result
    yield dict(
        border = dict(
            x = float(x),
            y = float(y),
            width = float(w),
            height = float(h),
            ),
            prediction = {'happiness': float(predicted_emotions[0]), 'neutral': float(predicted_emotions[1])},
            emotion = best_emotion
        )


def put_text(image, rect, text):
    x, y, w, h = rect

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 30.0
    font_thickness = int(round(font_scale * 1.5))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    center_text_x = x + (w // 2)
    center_text_y = y + (h // 2)
    text_w, text_h = text_size

    lower_left_text_x = center_text_x - (text_w // 2)
    lower_left_text_y = center_text_y + (text_h // 2)

    cv2.putText(
        image, text,
        (lower_left_text_x, lower_left_text_y),
        font, font_scale, (0, 255, 0), font_thickness
    )


def draw_face_info(image, face_info):
    x = int(face_info['border']['x'])
    y = int(face_info['border']['y'])
    w = int(face_info['border']['width'])
    h = int(face_info['border']['height'])
    emotion = face_info['emotion']

    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)
    put_text(image, (x, y, w, h // 5), emotion)


def show_image(image, title='Result'):
    # plt.subplot(111), plt.imshow(image), plt.title(title)
    # plt.show()
    pass


def cropEyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detect(gray, minimumFeatureSize=(80, 80))

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and
    #  right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(leftEye[1:3, 1])
    l_lowy = max(leftEye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye
    lw = (leftEye[3][0] - leftEye[0][0])

    # we want the image for the cnn to be (26,34)
    # so we add the half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    minxl = (leftEye[0][0] - ((34 - lw) / 2))
    maxxl = (leftEye[3][0] + ((34 - lw) / 2))
    minyl = (l_uppery - ((26 - l_dify) / 2))
    maxyl = (l_lowy + ((26 - l_dify) / 2))

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye at right eye
    r_uppery = min(rightEye[1:3, 1])
    r_lowy = max(rightEye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0] - ((34 - rw) / 2))
    maxxr = (rightEye[3][0] + ((34 - rw) / 2))
    minyr = (r_uppery - ((26 - r_dify) / 2))
    maxyr = (r_lowy + ((26 - r_dify) / 2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # resize for the conv net
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)
    # return left and right eye
    return left_eye_image, right_eye_image
# make the image to have the same format as at training
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

#
# if __name__ == '__main__':
#
#     # start time
#     start_time = time.time()
#     image, gray_image = load_image(sys.argv[1])
#
#     for face_info in predict(gray_image):
#         print(face_info)
#         draw_face_info(image, face_info)
#     # end time
#     end_time = time.time()
#     show_image(image)
#
#     response_time = end_time - start_time
#     print(response_time)
#
