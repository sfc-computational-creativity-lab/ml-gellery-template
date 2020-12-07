import cv2
from PIL import Image
import numpy as np
import hydra

from controller import Controller


@hydra.main('config.yml')
def main(cfg):

    def ratio(x):
        pass

    cv2.namedWindow('style transfer morphing')
    cv2.createTrackbar('alpha', 'style transfer morphing', 0, 100, ratio)

    cap = cv2.VideoCapture(0)

    controller = Controller(
        cfg.vgg_weight,
        cfg.decoder_weight,
    )
    style = Image.open(cfg.style)
    while True:
        alpha = cv2.getTrackbarPos('alpha', 'style transfer morphing') / 100

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        out = controller.transfer(img, style, alpha, cfg.resolution)
        out = np.array(out, dtype=np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imshow('style transfer', out)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
