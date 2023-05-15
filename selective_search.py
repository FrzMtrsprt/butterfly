import os
from typing import List

import cv2

from annotation_parser import BndBox

# speed-up using multithreads
cv2.setUseOptimized(True)
cpu_count = os.cpu_count()
cv2.setNumThreads(cpu_count if cpu_count else 1)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def selective_search(image_path: str) -> List[BndBox]:

    im = cv2.imread(image_path)

    # resize image
    new_h = 200
    scale_rate = new_h / im.shape[0]
    new_y = int(im.shape[1]*scale_rate)
    im = cv2.resize(im, (new_y, new_h))

    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    bndboxes: List[BndBox] = []
    for rect in rects:
        scaled_rect = (int(rect[0] / scale_rate),  # left
                       int(rect[1] / scale_rate),  # top
                       int(rect[2] / scale_rate),  # width
                       int(rect[3] / scale_rate))  # height
        bndbox = BndBox(xmin=scaled_rect[0], ymin=scaled_rect[1],
                        xmax=scaled_rect[0] + scaled_rect[2],
                        ymax=scaled_rect[1] + scaled_rect[3])
        bndboxes.append(bndbox)

    return bndboxes


if __name__ == '__main__':
    image_path = 'datasets/JPEGImages/IMG_000001.jpg'
    rects = selective_search(image_path)
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50
    im = cv2.imread(image_path)

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h),
                              (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        cv2.putText(imOut, str(numShowRects), (0, 100),
                    cv2.FONT_HERSHEY_PLAIN, 10, 1, 10)

        # show output
        cv2.imshow("Output", cv2.resize(imOut, (1920, 1080)))

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
