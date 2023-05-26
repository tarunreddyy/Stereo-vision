import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

###################################################################
##----------------------- Load dataset --------------------------##
###################################################################
def load_dataset(path):
    # Load images and calibration data
    data_path = path + 'calib.txt'
    with open(data_path, 'r') as file:
        lines = file.readlines()

    data = {}
    data['cam0'] = np.array([list(map(float, line.split())) for line in lines[0].split('=')[1].strip()[1:-1].split(';')])
    data['cam1'] = np.array([list(map(float, line.split())) for line in lines[1].split('=')[1].strip()[1:-1].split(';')])
    data['doffs'] = float(lines[2].split('=')[1].strip())
    data['baseline'] = float(lines[3].split('=')[1].strip())
    data['width'] = int(lines[4].split('=')[1].strip())
    data['height'] = int(lines[5].split('=')[1].strip())
    data['ndisp'] = int(lines[6].split('=')[1].strip())
    data['vmin'] = int(lines[7].split('=')[1].strip())
    data['vmax'] = int(lines[8].split('=')[1].strip())

    img0_file = path + 'im0.png'
    img1_file = path + 'im1.png'
    img0 = cv2.imread(img0_file)
    img1 = cv2.imread(img1_file)

    return img0, img1, data

###################################################################
##------------------------ Calibration --------------------------##
###################################################################
def feature_matching(img1, img2, max_points=800):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_points]

    pts1 = [kp1[m.queryIdx].pt for m in good_matches]
    pts2 = [kp2[m.trainIdx].pt for m in good_matches]

    return pts1, pts2

def plot_good_matches(img1, img2, pts1, pts2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:] = img2

    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]
        pt2_shifted = pt2 + np.array([w1, 0])

        ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 'r-')
        ax.plot(pt1[0], pt1[1], 'bo')
        ax.plot(pt2_shifted[0], pt2_shifted[1], 'go')

    ax.imshow(combined_img)
    plt.show()

def RANSAC(norm_pts1, norm_pts2):
    n = norm_pts1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [
            norm_pts1[i, 0] * norm_pts2[i, 0],
            norm_pts1[i, 0] * norm_pts2[i, 1],
            norm_pts1[i, 0],
            norm_pts1[i, 1] * norm_pts2[i, 0],
            norm_pts1[i, 1] * norm_pts2[i, 1],
            norm_pts1[i, 1],
            norm_pts2[i, 0],
            norm_pts2[i, 1],
            1
        ]
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    return np.dot(U, np.dot(np.diag(S), V))

def normalize_points(points):
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)

    T = np.array([
        [1 / std_dev[0], 0, -mean[0] / std_dev[0]],
        [0, 1 / std_dev[1], -mean[1] / std_dev[1]],
        [0, 0, 1]
    ])

    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    norm_points = np.dot(T, points_h.T).T[:, :2]

    return norm_points, T

def estimate_fundamental_matrix(pts1, pts2, max_iterations=2000, threshold=1e-3):
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)
    
    best_inliers = 0
    best_fundamental_matrix = None

    for _ in range(max_iterations):
        random_indices = np.random.choice(norm_pts1.shape[0], 8, replace=False)
        random_norm_pts1 = norm_pts1[random_indices]
        random_norm_pts2 = norm_pts2[random_indices]

        F = RANSAC(random_norm_pts1, random_norm_pts2)

        norm_pts1_h = np.hstack((norm_pts1, np.ones((norm_pts1.shape[0], 1))))
        norm_pts2_h = np.hstack((norm_pts2, np.ones((norm_pts2.shape[0], 1))))
        error = np.abs(np.sum(norm_pts2_h * np.dot(norm_pts1_h, F.T), axis=1))

        inliers = error < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_fundamental_matrix = F

    F_denorm = np.dot(T2.T, np.dot(best_fundamental_matrix, T1))
    return F_denorm / F_denorm[2, 2]

def find_essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U,S,V = np.linalg.svd(E)
    #rank constraint
    S = [1,1,0]
    new_E = np.dot(U,np.dot(np.diag(S),V))
    return new_E

def Decompose_essential_matrix(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    Rots = [np.dot(U, np.dot(W, V_T)), np.dot(U, np.dot(W.T, V_T))]
    trans = [U[:, 2], -U[:, 2]]

    possible_poses = []
    for R in Rots:
        for t in trans:
            if np.linalg.det(R) > 0:
                possible_poses.append((R, t))

    return possible_poses

def triangulate_points(P1, P2, pts1, pts2):
    pts_3D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3D /= pts_3D[3]
    return pts_3D[:3]

def check_cheirality(R, C, pts_3D):
    R3 = R[2].reshape(1, -1)
    C = C.reshape(3, 1) # Ensure C has a shape of (3, 1)
    num_positive = np.sum(np.dot(R3, pts_3D - C) > 0)
    return num_positive

def recover_camera_pose(E, K1, K2, pts1, pts2):
    possible_poses = Decompose_essential_matrix(E)
    best_pose = None
    max_positive = 0

    for R, C in possible_poses:
        P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(K2, np.hstack((R, C.reshape(-1, 1))))
        pts_3D = triangulate_points(P1, P2, pts1, pts2)
        num_positive = check_cheirality(R, C, pts_3D)

        if num_positive > max_positive:
            max_positive = num_positive
            best_pose = (R, C, pts_3D)

    return best_pose

###################################################################
##------------------------ Rectification-------------------------##
###################################################################
def draw_epipolar_lines(image, lines, points):
    epipolar_img = image.copy()
    for i in range(points.shape[0]):
        pt = points[i]
        y = int(pt[1])
        x_min, x_max = 0, image.shape[1]
        cv2.circle(epipolar_img, (int(pt[0]), y), 10, (0, 0, 255), -1)
        epipolar_img = cv2.line(epipolar_img, (x_min, y), (x_max, y), (0, 255, 0), 2)
    return epipolar_img


def rectify_and_compute_epi_lines(image0, image1, F, pts_set1, pts_set2):
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts_set1), np.float32(pts_set2), F, imgSize=(w1, h1))

    img_rect = []
    for img, H, w, h in ((image0, H1, w1, h1), (image1, H2, w2, h2)):
        img_rectified = cv2.warpPerspective(img, H, (w, h))
        img_rect.append(img_rectified)

    pts_set_rect = []
    for pts, H in ((pts_set1, H1), (pts_set2, H2)):
        pts_rectified = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        pts_set_rect.append(pts_rectified)

    H2_T_inv, H1_inv = np.linalg.inv(H2.T), np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F, H1_inv))

    lines = []
    for i, pts in enumerate(pts_set_rect):
        lines.append(np.dot(np.dot(F_rectified, np.hstack((pts, np.ones((pts.shape[0], 1)))).T).T, (H1 if i == 0 else H2.T)))

    epipolar_img = []
    for img, line, pts in zip(img_rect, lines, pts_set_rect):
        epipolar_img.append(draw_epipolar_lines(img, line, pts))

    img_group = np.concatenate(epipolar_img, axis=1)

    plt.figure(figsize=(15,10))
    plt.imshow(img_group)

    return img_rect[0], img_rect[1], pts_set_rect, F_rectified


###################################################################
##------------------------ Correspondence -----------------------##
###################################################################
def compute_disparity_map(img_rect_left, img_rect_right, window = 5):
    height, width = img_rect_left.shape[:2]
    disparity_map = np.zeros((height, width))
    x_new = width - (2 * window)

    for y in tqdm(range(window, height - window)):
        block_img_rect_left, block_img_rect_right = [], []
        for x in range(window, width - window):
            block_left = img_rect_left[y:y + window, x:x + window].flatten()
            block_right = img_rect_right[y:y + window, x:x + window].flatten()

            block_img_rect_left.append(block_left)
            block_img_rect_right.append(block_right)

        block_img_rect_left = np.array(block_img_rect_left)
        block_img_rect_right = np.array(block_img_rect_right)

        block_img_rect_left = np.repeat(block_img_rect_left[:, :, np.newaxis], x_new, axis=2)
        block_img_rect_right = np.repeat(block_img_rect_right[:, :, np.newaxis], x_new, axis=2)

        block_img_rect_right = block_img_rect_right.T

        squared_difference = np.square(block_img_rect_left - block_img_rect_right)
        SSD = np.sum(squared_difference, axis=1)
        index = np.argmin(SSD, axis=0)
        disparity = np.abs(index - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity

    return disparity_map


###################################################################
##--------------------- Compute Depth Image ---------------------##
###################################################################
def plot_disparity_and_depth(disparity_map, baseline, f, data_set_name):
    depth_map = baseline * f / (disparity_map + 1e-8)
    valid_mask = np.where(disparity_map > 0, 1, 0)
    depth_map *= valid_mask

    disparity_map_int = (disparity_map / np.max(disparity_map) * 255).astype(np.uint8)
    depth_map_int = (depth_map / np.max(depth_map) * 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    axs[0, 0].set_title('Disparity Map (Grayscale)')
    axs[0, 1].imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    axs[0, 1].set_title('Disparity Map (Color)')
    axs[1, 0].imshow(depth_map_int, cmap='gray', interpolation='nearest')
    axs[1, 0].set_title('Depth Map (Grayscale)')
    axs[1, 1].imshow(depth_map_int, cmap='hot', interpolation='nearest')
    axs[1, 1].set_title('Depth Map (Color)')
    for ax in axs.flat:
        ax.set_axis_off()

    plt.savefig(f'results/disparity_depth_{data_set_name}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


###################################################################
##------------------- Pipeline Implementation -------------------##
###################################################################
def stereo_vision_pipeline(path, dataset_name):

    img0, img1, calib_data = load_dataset(path)
    img0_pts, img1_pts = feature_matching(img0, img1)
    # plot_good_matches(img0, img1,img0_pts,img1_pts)
    fundamental_matrix = estimate_fundamental_matrix(np.array(img0_pts), np.array(img1_pts))
    print(f"Fundamental Matrix = {fundamental_matrix}")
    E = find_essential_matrix(fundamental_matrix, calib_data["cam0"], calib_data["cam1"])
    print(f"Essential Matrix = {E}")
    R, T, pts_3D = recover_camera_pose(E, calib_data["cam0"], calib_data["cam1"], np.array(img0_pts), np.array(img1_pts))
    print(f"Rotation Matrix = {R}\n Translation Matrix = {T}")
    img_rect_left, img_rect_right, pts_set_rect, F_rectified = rectify_and_compute_epi_lines(img0, img1, fundamental_matrix, np.array(img0_pts), np.array(img1_pts))
    disparity_map = compute_disparity_map(img_rect_left, img_rect_right, 3)
    plot_disparity_and_depth(disparity_map, calib_data["baseline"], calib_data["cam0"][0,0], dataset_name)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Run stereo vision on a dataset')
        parser.add_argument('--dataset', dest='dataset', type=str, required=True, choices=['artroom', 'chess', 'ladder'], help='Dataset to use (artroom, chess, ladder)')
        args = parser.parse_args()

        if args.dataset == 'artroom':
            stereo_vision_pipeline('datasets/artroom/', args.dataset)
        elif args.dataset == 'chess':
            stereo_vision_pipeline('datasets/chess/', args.dataset)
        elif args.dataset == 'ladder':
            stereo_vision_pipeline('datasets/ladder/', args.dataset)
    except:
        print("**Error: Run the code again!! could not compute R and T")