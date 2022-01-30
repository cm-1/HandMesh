import cv2
import numpy as np

import runnerCreator

class webcamManager:
    def __init__(self, focalLength, width, height):
        cap = cv2.VideoCapture(0) #,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap = cap
        refWidthForFocalLength = max(width, height)
        self.runner = runnerCreator.getRunner(focalLength, refWidthForFocalLength)

        K = np.identity(3)
        K[0, 0] = focalLength
        K[1, 1] = focalLength
        K[0, 2] = width // 2
        K[1, 2] = height // 2
        self.displayK = K

    def displayFrame(self):
        ret, frame = self.cap.read()

        convertedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            return

        handInfo = self.runner.processFrame(convertedFrame)
        colourList = [(0, 0, 255), (0, 255, 0)]
        # Iterate over all hands detected.
        for handNum in range(handInfo.points2D.shape[0]):
            # Iterate over all points detected for said hand.
            for landmarkNum in range(handInfo.points2D.shape[1]):
                pt = (handInfo.points2D[handNum][landmarkNum]).astype(int)

                circleRadius = 5
                colour = colourList[handNum]
                thickness = 1
                
                frame = cv2.circle(frame, pt, circleRadius, colour, thickness)

        # Iterate over all outputed 3D points:
        for pointNum in range(handInfo.points3D.shape[0]):
            pt3D = handInfo.points3D[pointNum]
            projected = np.matmul(self.displayK, pt3D)
            pt2D = (projected[0]/projected[2], projected[1]/projected[2])
            pt2D_int = (int(pt2D[0]), int(pt2D[1]))
        
            circleRadius = 5
            colour = (255, 255, 0)
            thickness = 1
            
            frame = cv2.circle(frame, pt2D_int, circleRadius, colour, thickness)

        cv2.imshow("test", frame)



    

print("Starting")
w = 640
h = 480
focalLength = 634
wcm = webcamManager(focalLength, w, h)
while True:
    wcm.displayFrame()
    # Break on press of "Q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Done")

