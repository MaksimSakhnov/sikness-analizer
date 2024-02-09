import os
import cv2


groups = {
    1: 'bottom_left',
    2: 'bottom_right',
    3: 'top_left',
    4: 'top_right',
}

def subtract_images():
    result1 = 1
    result2 = 1
    result3 = 1
    result4 = 1
    for group in groups:
        gr = groups[group]
        background = cv2.imread(f'spliced_video/{gr}_{0}.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
        for i in range(2, 925):
            cur_image = cv2.imread(f'spliced_video/{gr}_{i}.jpg', cv2.IMREAD_GRAYSCALE)
            subtracted = cv2.absdiff(cur_image.astype(float), background)
            _, thresholded = cv2.threshold(subtracted, 25, 255, cv2.THRESH_BINARY)
            cur_res = cv2.dilate(thresholded, None, iterations=2)
        if (group == 1):
            result1 = cur_res
        elif(group == 2):
            result2 = cur_res
        elif (group == 3):
            result3 = cur_res
        elif (group == 4):
            result4 = cur_res
        output_dir = 'result_photo'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(f'{output_dir}/result1.png', result1)
        cv2.imwrite(f'{output_dir}/result2.png', result2)
        cv2.imwrite(f'{output_dir}/result3.png', result3)
        cv2.imwrite(f'{output_dir}/result4.png', result4)




