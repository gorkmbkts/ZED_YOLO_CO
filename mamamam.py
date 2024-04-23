import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pyzed.sl as sl
import ogl_viewer.tracking_viewer as gl
import math
import os
from threading import Lock, Thread
from time import sleep
lock = Lock()
run_signal = False
exit_signal = False

# Her şeyden önce Python nazar duası ve ASM tanrısından merhamet dileme
"""
Oh, glorious ASM gods, both benevolent and buggy,
Hear my plea, this program I beseech thee.
Keep the gremlins at bay, the errors unseen,
And grant my logic a path ever clean.

May no stack overflow crash my grand plan,                  [amiiiin]
Nor infinite loops drive me mad, like a banned fan.
Let functions fire true, variables hold tight,              [amiiiin]
And algorithms dance in the silicon light.

From the depths of the web, where bugs congregate,
Deliver us, oh ASM gods, from their mischievous fate.       [amiiiin]
Shield us from spiders, their webs all a mess,              [amiiiin]
And fruit flies who dare test our keyboard finesse.

Grant us patience, for bugs will arise,
But guide us, oh ASM gods, to squash them with wise eyes.
With debugging tools sharp, and logic refined,
We'll conquer the errors, one line at a time.

So bless this endeavor, this code we create,
And banish the demons of bugs and their hate.
With humor and hope, we embark on this quest,
May your compiler's blessing put all errors to rest.

Amen and, optionally, a healthy dose of caffeine
"""

print("ZED Baslatiliyor...")
    
zed = sl.Camera()
input_type = sl.InputType()

init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # fastest
#init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # quality
init_params.coordinate_units = sl.UNIT.METER # Use millimeter units

runtime_params = sl.RuntimeParameters()
status = zed.open(init_params)
camera_infos = zed.get_camera_information()
camera_res = camera_infos.camera_configuration.resolution

## point cloud and depth initialiasion (kesin yanlış yazdım) ##
point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
point_cloud_render = sl.Mat()
point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
depth = sl.Mat()


def AkilliZeka(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections,frame,model

    print("Neural Network Baslatiliyor...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()
            #threadlerin globalde aynı senkron data kullanabilmesi için verilerin eşitlenmeini beklemeye başlıyoruz
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            #zed'den gelen 4 channel streami 3 channela düşürme
            for result in model.track(img, show=False, stream=True, agnostic_nms=True,verbose=False, persist=True ):
                frame = result.orig_img
                #model.track'ın içerisinde orig_img verisini çekerek yolo ile paralel işlenmiş görüntüyü elde ediyoruz
                detections = sv.Detections.from_yolov8(result)
                #supervision kullanarak Detections objesi oluşturuyoruz, internette çok az bilgi var kaynak kodu okumak gerekiyor 
                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    #Detections'un boş olup olmadığını kontrol ediyoruz. Olmayan objeye cpu'dan bellek ayırırsan patlıyacaktır
                #kordinatverlan(detections[(detections.class_id == 0)])
                kordinatverlan(detections)
                #sadece insan detecttrack ayarladı, tümünü almak için id bağlamını kaldı
                
                ########################## Supervision Annotators ##############################
                box_annotator = sv.BoxAnnotator(
                    thickness=2,
                    text_thickness=1,
                    text_scale=0.5
                )
                labels = [
                        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                        for _, confidence, class_id, tracker_id
                        in detections
                    ]
                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections,
                    labels=labels
                )
                ###############################################################################
            lock.release()
            #tüm veriler bir kare ve her detection için hesaplandı. veri eşitlenmesi için beklemeyi bırakıyoruz
            run_signal = False
        sleep(0.01)

def kordinatverlan(detections):
    os.system("cls")    
    #terminalde okunabilirlik için her frame sonrası terminal temizleniyor    
    try:
    #detectionun boş olup olmadığını burada kontrol edilmesi mümkün olmadığı için try/exc kullanmak durumundayız
    #sonrasında bu yöntem daha iyi bir yöntemle değiştirilebilir    
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            #indeksteki detectionun xyxy format kordinatları alınıyor
            Xcen=((x1+x2)/2).astype(int)
            Ycen=((y1+y2)/2).astype(int)
            #her detectionun orta noktası bulunuyor
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
            ortDepth=0
            #zed kameradan point cloud ve depth ölçümleri alınıyor, bu aşamada point cloud kullanılmıyor ancak daha sonra doğrulama için kullanılabilir
            
            for i in range(10):
               for j in range(10):
                    status, depth_value = depth.get_value(Xcen+i-5, Ycen+j-5)
                    ortDepth += depth_value

            ortDepth=ortDepth/100        
                                
            #merkez noktadan derinlik değeri alınıyor                
            print(f"//////////////////////////////////////")
            print(f"classid: {detections.class_id[detection_idx]}  |  trackid: {detections.tracker_id[detection_idx]}")
            print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
            print(f"Center of the {model.model.names[detections.class_id[detection_idx]]} ===> x:{Xcen} y:{Ycen}")
            print(f"Distance from ZED to center of the {model.model.names[detections.class_id[detection_idx]]} ===> {depth_value} mm")
            print(f"Average 10x10 distance from ZED to center of the {model.model.names[detections.class_id[detection_idx]]} ===> {ortDepth} mm")
            print("//////////////////////////////////////\n\n\n\n\n")           
    except Exception as err:
        pass
        #print("there is no object that YOLO can detect"  , end="\r")
        #print(f"Unexpected {err=}, {type(err)=}")

def main():
    global image_net, exit_signal, run_signal, detections,frame,model,camera_res
    
    Ai_thread = Thread(target=AkilliZeka, kwargs={'weights': 'yolov8l.pt', 'img_size':416, "conf_thres": 0.4})
    Ai_thread.start()
    image_left_tmp = sl.Mat()
    image_left = sl.Mat()  
    #ZED kameranın sol lensinden görüntü alacağız bu almadan önce sl.Mat Variable oluşturuyoruz
  
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_imu_fusion = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    #positional tracking için gerekli ayarlamalar...

    camera_info = zed.get_camera_information()
    viewer = gl.GLViewer()
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)
    viewer.init(camera_info.camera_model)
    sensors_data = sl.SensorsData()
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    #tracking viewer için ayarlamalar...  
    
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
    #2D görüntü için ayarlamalar...

    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    cam_w_pose = sl.Pose()
    #tracking viewer için ayarlamal...  


    text_translation = ""
    text_rotation = ""
    file = open('output_trajectory.csv', 'w')
    file.write('tx, ty, tz \n')
    #tracking output trajectory verilerini yazmak için bir .csv dosyası oluşturuluyor

    key = ' '
    while key != 113 and viewer.is_available():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            tracking_state = zed.get_position(cam_w_pose,sl.REFERENCE_FRAME.WORLD)
            # tracking state alınıyor

            # img alınıyor
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # Detectionlar aynı zamanda hesaplanıyor
            while run_signal:
                sleep(0.001)
            # Detectionların tamamlanmasını bekle
            """    
            lock.acquire()
            
            lock.release()
            """
            zed.retrieve_objects(objects, obj_runtime_param)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    rotation = cam_w_pose.get_rotation_vector()
                    translation = cam_w_pose.get_translation(py_translation)
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                    pose_data = cam_w_pose.pose_data(sl.Transform())
                    file.write(str(translation.get()[0])+", "+str(translation.get()[1])+", "+str(translation.get()[2])+"\n")
               
            viewer.updateData(pose_data, text_translation, text_rotation, tracking_state)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
            #Tracker hesaplamaları ve veri updateleri


            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
           
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            
            
            cv2.imshow("yolov8", frame)
            
            
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

   
    exit_signal = True
    viewer.exit()
    zed.close()

if __name__ == "__main__":
    main()
    
