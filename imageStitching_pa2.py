import numpy as np
import cv2
import os

def load(folder):
    images=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def getHomography(first_image, second_image, algorithm):

    if (algorithm == "SIFT"):
        #SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(first_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(second_image, None)

    elif ( algorithm == "ORB"):
        #ORB #cant detect matches
        orb = cv2.ORB_create()
        keypoints1= orb.detect(first_image, None)
        keypoints2 = orb.detect(second_image, None)
        keypoints1, descriptors1 = orb.compute(first_image, keypoints1)
        keypoints2, descriptors2 = orb.compute(second_image, keypoints2)

    elif ( algorithm == "SURF "):
        SURF
        surf = cv2.xfeatures2d.SURF_create()
        keypoints1, descriptors1 = surf.detectAndCompute(first_image, None)
        keypoints2, descriptors2 = surf.detectAndCompute(second_image, None)


    #showing keypoints
    #first_image_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    #img_1 = cv2.drawKeypoints(first_image_gray, keypoints1, first_image)
    #cv2.imshow("keypoints.jpg", img_1)
    #cv2.waitKey(0)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    #showing matches
    #img3 = cv2.drawMatches(first_image, keypoints1,second_image, keypoints2, good_matches[:32], outImg= None)
    #cv2.imshow("matches.jpg", img3)
    #cv2.waitKey(0)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]) \
        .reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]) \
        .reshape(-1, 1, 2)
    if len(src_pts) > 5:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    else:
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return M


def blend(img1, img2, mask, level):
    n1 = img1.astype(np.float64)
    n2 = img2.astype(np.float64)
    m = mask.astype(np.float64)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(3):
                if m[i,j,k] == 1 and img2[i,j,k] > 0:
                    m[i,j,k] = 0.5

    #Generate Gaussian Pyramid
    n1_copy = n1.copy()
    gp_n1 = [n1_copy]

    n2_copy = n2.copy()
    gp_n2 = [n2_copy]

    m_copy = m.copy()
    gp_m = [m_copy]

    for i in range(level):
        n1_copy = cv2.pyrDown(n1_copy)
        gp_n1.append(n1_copy)

        n2_copy = cv2.pyrDown(n2_copy)
        gp_n2.append(n2_copy)

        m_copy = cv2.pyrDown(m_copy)
        gp_m.append(m_copy)

    #Laplacian Pyramid
    lp_n1 = [gp_n1[level-1]]
    lp_n2 = [gp_n2[level-1]]
    lp_m = [gp_m[level-1]]

    for i in range(level-1, 0, -1):
        n1_ex = cv2.pyrUp(gp_n1[i])
        lap1 = cv2.subtract(gp_n1[i-1], n1_ex)
        lp_n1.append(lap1)

        n2_ex = cv2.pyrUp(gp_n2[i])
        lap2 = cv2.subtract(gp_n2[i - 1], n2_ex)
        lp_n2.append(lap2)

        lp_m.append(gp_m[i-1])

    #Join Half the Pyramid
    n1_n2_pyramid = []
    for n1_lap, n2_lap, gm in zip(lp_n1, lp_n2, lp_m):
        ls = n1_lap * gm + n2_lap * (1.0 - gm)
        n1_n2_pyramid.append(ls)

    #Reconstruct
    reconstruct = n1_n2_pyramid[0]
    for i in range(1, level):
        reconstruct = cv2.pyrUp(reconstruct)
        reconstruct = cv2.add(reconstruct, n1_n2_pyramid[i])

    return reconstruct


def warpImage(middle, left, right, level, border1, border2, algorithm ):
    middle = cv2.copyMakeBorder(middle, 0, 0, border1, border2, cv2.BORDER_CONSTANT)
    homography1 = getHomography(left,middle, algorithm)
    homography2 = getHomography(right,middle, algorithm)

    height, width, channels = middle.shape

    try:
        left_perspective = cv2.warpPerspective(left, homography1, (width, height))  # sol warped
        m1 = np.ones_like(left, dtype='float32')
        left_mask = cv2.warpPerspective(m1, homography1, (width, height))  # sol mask

        right_perspective = cv2.warpPerspective(right, homography2, (width,height)) #sag warped
        m2 = np.ones_like(right, dtype='float32')
        right_mask = cv2.warpPerspective(m2, homography2, (width, height)) #sag mask

        blend1 = blend(left_perspective, middle, left_mask, level)
        blend2 = blend(right_perspective, blend1, right_mask, level)

        return blend2
    except:
        return middle


def panoramic(middle, mid, border1, i, algorithm):
    middle = warpImage(middle, images[mid-i], images[mid+i], 1 , border1, border1, algorithm)
    middle = np.clip(middle, 0, 255).astype(np.uint8)
    i+=1
    if i == mid:
        return middle
    return panoramic(middle, mid, border1, i, algorithm)


#CHOOSE DATASET HERE
path_folder = r"D:\2022-BAHAR\416\ass2\int\cvc01passadis-cyl-pano21"
os.chdir(path_folder)
images = load(path_folder)
images = images[6:32] #delete images that corrupts the warping
level=1


alg = "SIFT"
midd= int(len(images)/2)
mid_start = warpImage(images[midd], images[midd-1], images[midd+1], 1 , 150, 150, alg)
mid_start = np.clip(mid_start,0,255).astype(np.uint8)
final = panoramic(mid_start, midd, 100, 1, alg)

cv2.imwrite("panoramic_SIFT.jpg", final)


alg = "ORB"
midd= int(len(images)/2)
mid_start = warpImage(images[midd], images[midd-1], images[midd+1], 1 , 150, 150, alg)
mid_start = np.clip(mid_start,0,255).astype(np.uint8)
final = panoramic(mid_start, midd, 100, 1, alg)

cv2.imwrite("panoramic_ORB.jpg", final)


