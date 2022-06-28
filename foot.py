import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


SHOE_SIZE_CONVERSION_BY_LENGTH_TABLE = {
    24.4: {"US": 7, "UK": 6, 'VN': 40},
    24.8: {'US': 7.5, "UK": 6.5, "VN": 40.5},
    25.2: {"US": 8, "UK": 7, "VN": 41},
    25.7: {"US": 8.5, "UK": 7.5, "VN": 41.5},
    26: {"US": 9, "UK": 8, "VN": 42},
    26.5: {"US": 9.5, "UK": 8.5, "VN": 42.5},
    26.8: {"US": 10, "UK": 9, "VN": 43},
    27.3: {"US": 10.5, "UK": 9.5, "VN": 43.5},
    27.8: {"US": 11, "UK": 10, "VN": 44},
    28.3: {"US": 11.5, "UK": 10.5, "VN": 44.5},
    28.6: {"US": 12, "UK": 11, "VN": 45},
    29.4: {"US": 13, "UK": 12, "VN": 46}
}


SHOE_SIZE_CONVERSION_BY_WIDTH_TABLE = {
    9.8: {"US": 5, "UK": 4.5, "VN": 38},
    10: {"US": 6, "UK": 5.5, "VN": 39},
    10.2: {"US": 7, "UK": 6.5, "VN": 40},
    10.4: {"US": 8, "UK": 7.5, "VN": 41},
    10.6: {"US": 9, "UK": 8.5, "VN": 42},
    10.8: {"US": 10, "UK": 9.5, "VN": 43},
    11: {"US": 11, "UK": 10.5, "VN": 44}
}


def segment(img, num_clusters, method='gmm'):
    """
    Segment image
    :param img: Image to be segmented
    :param num_clusters: The number of clusters
    :param method: can be either 'kmeans' (K-means) or 'gmm' (Gaussian Mixture Models)
    :return: Segmented image
    """
    # image needs to be flatten to 2D to use K-means
    img2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    segmented_img = None
    if method == 'kmeans':
        segmentor = KMeans(n_clusters=num_clusters, random_state=1).fit(img2D)
        segmented_img = segmentor.cluster_centers_[segmentor.labels_]
    elif method == 'gmm':
        segmentor = GaussianMixture(n_components=num_clusters, random_state=1).fit(img2D)
        segmented_img = segmentor.means_[segmentor.predict(img2D)]
    # reshape the image back to its original shape
    segmented_img = segmented_img.reshape(img.shape)
    # convert back to 8 bit values
    segmented_img = np.uint8(segmented_img)
    # blur, erode then dilate to reduce noise
    segmented_img = cv2.medianBlur(segmented_img, 19)
    segmented_img = cv2.erode(segmented_img, np.ones((7, 7), np.uint8), iterations=5)
    segmented_img = cv2.dilate(segmented_img, np.ones((7, 7), np.uint8), iterations=5)
    return segmented_img


def detect_edge(segmented_img):
    """
    Detect edge from segmentation
    :param segmented_img: Segmentation of the original image
    :return: Edge image
    """
    edge_img = cv2.Canny(segmented_img, 100, 200)
    edge_img = cv2.dilate(edge_img, np.ones((7, 7), np.uint8), iterations=1)
    edge_img = cv2.erode(edge_img, np.ones((7, 7), np.uint8), iterations=1)
    return edge_img


def get_bounding_box(edged_img, original_image=None):
    """
    This function calculate the contour and bounding boxes for an edged image.
    :param edged_img: Image containing only the edge of the original image
    :param original_image: If provided, this function will draw the contours and bounding boxes on this og image
    :return: Bounding box of the largest contour, and box points of said box
    """
    contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_smooth = []
    # bounding_boxes = []
    boxes = []
    rect = []
    for i, c in enumerate(contours):
        contours_smooth.append(cv2.approxPolyDP(c, 3, True))
        # bounding_boxes.append(cv2.boundingRect(contours_smooth[i]))
        boxes.append(cv2.minAreaRect(contours_smooth[i]))
        rect.append(cv2.boxPoints(boxes[i]))
        rect[i] = np.int0(rect[i])
    # bounding_boxes.sort(key=lambda x: x[2]*x[3], reverse=True)
    # boxes.sort(key=lambda x: x[1][0]*x[1][1], reverse=True)
    # rect.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    if original_image is not None:
        cv2.drawContours(original_image, contours_smooth, -1, (0, 255, 0), 3)
        cv2.drawContours(original_image, [rect[0]], 0, (255, 0, 0), 3)
    # for box in bounding_boxes[0:]:
    # cv2.rectangle(orig, (box[0], box[1], box[0]+box[2], box[1]+box[3]), (255, 0, 0), 3)
    return boxes[0], rect[0]


def min_max_scale(img):
    mx = img.max()
    mi = img.min()
    return (img - mi) / (mx - mi)


def laplacian_sharper(img):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    return (min_max_scale(-min_max_scale(imgLaplacian) + min_max_scale(img))*255).astype('uint8')


def convert_to_shoe_size(size, measure='length'):
    shoe_size = None
    if measure == 'length':
        size += 1.5
        for ref in SHOE_SIZE_CONVERSION_BY_LENGTH_TABLE.keys():
            if size >= ref:
                shoe_size = SHOE_SIZE_CONVERSION_BY_LENGTH_TABLE[ref]
            else:
                return shoe_size
    elif measure == 'width':
        for ref in SHOE_SIZE_CONVERSION_BY_WIDTH_TABLE.keys():
            if size >= ref:
                shoe_size = SHOE_SIZE_CONVERSION_BY_WIDTH_TABLE[ref]
            else:
                return shoe_size
    return
