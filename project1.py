from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from zone_detector import ZoneDetector
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

zone = [
    (320,20),
    (620,20),
    (620,600),
    (320,600)
]

zone_detector = ZoneDetector(
        "Zone_A",
        zone)

zone_points = np.array(zone,np.int32)

cap = cv2.VideoCapture("more.mp4")
if not cap.isOpened():
    print("ERROR: Could not open people.mp4")
    exit()


frame_count = 0
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break

    frame_count += 1

    if frame_count %2 != 0:
        continue

    # YOLO detection
    results = model(frame, verbose= False,imgsz =320)

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            width = x2 - x1
            height = y2 - y1

            detections.append(
                ([x1, y1, width, height],
                 confidence,
                 "person")
            )

    # DeepSORT
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # Draw tracking boxes
    persons_inside = 0
    for track in tracks:

        if not track.is_confirmed():
            continue

        if track.time_since_update > 0:
            continue

        track_id = track.track_id

        x1, y1, x2, y2 = map(
            int,
            track.to_ltrb()
        )
        zone_result = zone_detector.check_person(
                track_id,
                x1,
                y1,
                x2,
                y2
        )

        status = zone_result["zone_status"]

        if status =="INSIDE_ZONE":
            persons_inside += 1


        if status == "INSIDE_ZONE":
            box_color = (0,255,0)
        elif status == "OUTSIDE_ZONE":
            box_color = (0,0,255)

        else:
            box_color = (0,255,255)


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1,max(y1-35,20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
            
        )

        cv2.putText(
                frame,
                status,
                (x1,max(y1 - 10,20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
        )






        if "event" in zone_result:
            print(zone_result)

    overlay = frame.copy()
    cv2.fillPoly(
            overlay,
            [zone_points],
            (255,0,0)
    )

    frame = cv2.addWeighted(
            overlay,
            0.15,
            frame,
            0.85,
            0
    )


    cv2.polylines(
        frame,
        [zone_points],
        True,
        (255,0,0),
        3
    )


    cv2.rectangle(
        frame,
        (10,10),
        (380,65),
        (0,0,0),
        -1
    )

    cv2.putText(
            frame,
            f"{persons_inside} persons inside the zone",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
    )




    # Show video
    cv2.imshow("YOLO + DeepSORT", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break

cap.release()
cv2.destroyAllWindows()

print("Done")
