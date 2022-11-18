package com.lg2.beney.augmented_video.core;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerCSRT;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@SuppressWarnings("unused")
public class ObjectDetector {

    private static final String LOG_TAG = ObjectDetector.class.getSimpleName();

    private final Context mContext;


    ///////////////////////////////////////////////////////////////////////////
    // Tracker Features
    private static final float AREA_CHANGED_THRESHOLD = 0.2f;
    private static final float RATIO_CHANGED_THRESHOLD = 0.2f;
    private static final float WH_RATIO_THRESHOLD = 1.4f;
    private static final float MOVE_THRESHOLD = 0.05f;
    private static final boolean HISTORICAL_BOUNDARY = true;
    private static final int HISTORICAL_BOUNDARY_MAX_TRY = 150;

    // Detect Values
    private Tracker mTracker;
    private boolean mIsTracking;
    private int mTrackingFailCount;
    private Rect2d mTrackingObjectLocation = new Rect2d();
    private final float mResizeFx;
    private final float mResizeFy;

    ///////////////////////////////////////////////////////////////////////////
    // Detect Features
    private static final int INPUT_WIDTH = 416;
    private static final int INPUT_HEIGHT = 416;
    private static final float RESIDUAL_BOUNDARY_RATIO = 1.1f;
    private static final float CONFIDENCE_THRESHOLD = 0.4f;
    private static final List<String> TRACKING_OBJECT_LABELS = Collections.singletonList("tvmonitor");

    // Detect Values
    private final ArrayList<String> mClasses = new ArrayList<>();
    private static final String CLASSES_FILENAME = "yolov3n.names";
    private static final String MODEL_CONFIGURATION_FILENAME = "/yolov3c.cfg.gif";
    private static final String MODEL_WEIGHTS_FILENAME = "/yolov3w.weights.gif";

    private final Net mNet;
    private final List<String> mOutputLayers;

    private final DecimalFormat mFrameIndexDf = new DecimalFormat("00000000");

    public ObjectDetector(Context context) {
        this(context, 0.4f, 0.4f);
    }

    public ObjectDetector(Context context, float resizeFx, float resizeFy) {

        log("Constructor");
        log("ExternalStoragePath=" + Environment.getExternalStorageDirectory().getPath());

        mContext = context;

        mIsTracking = false;
        mTrackingFailCount = 0;
        mResizeFx = resizeFx;
        mResizeFy = resizeFy;

        String externalStoragePath = Environment.getExternalStorageDirectory().getPath();
        String modelConfiguration = externalStoragePath + MODEL_CONFIGURATION_FILENAME;
        String modelWeights = externalStoragePath + MODEL_WEIGHTS_FILENAME;

        readClasses(mClasses);
        mNet = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        mNet.setPreferableBackend(Dnn.DNN_BACKEND_VKCOM);
        mNet.setPreferableTarget(Dnn.DNN_TARGET_VULKAN);

        mOutputLayers = getOutputsNames(mNet);
    }

    private void readClasses(ArrayList<String> classes){

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(mContext.getAssets().open(CLASSES_FILENAME)))) {

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes.add(mLine);
            }
        } catch (IOException e) {
            //log the exception
        }
        //log the exception
    }

    private List<String> getOutputsNames(Net net)
    {
        ArrayList<String> names = new ArrayList<>();

            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
            //get the names of all the layers in the network
            List<String> layersNames = net.getLayerNames();

            // Get the names of the output layers in names
            for (Integer i : outLayers) {
                names.add(layersNames.get(i - 1));
            }

        return names;
    }

    public Rect run(Mat frame, int frameIndex, boolean doObjectDetection) {

        // Image 로드
        Mat inputFrame = new Mat();
        Imgproc.resize(frame, inputFrame, new Size(0,0), mResizeFx, mResizeFy);
        int height = inputFrame.height(), width = inputFrame.width();

        // Step 1. 트래킹 중이면 새로운 Image 에서 트래킹 중인 Object 를 찾는다
        if (mIsTracking) {

            Rect newObjectLocation = trackObject(inputFrame, frameIndex);

            // Tracking 성공
            if (newObjectLocation != null) {
                return toRect(
                        newObjectLocation.x / mResizeFx,
                        newObjectLocation.y / mResizeFy,
                        newObjectLocation.width / mResizeFx,
                        newObjectLocation.height / mResizeFy);
            }
            // Tracking 실패
            else {
                mIsTracking = false;
            }
        }

        // Should NOT be passed here.
        //noinspection ConstantConditions
        if (mIsTracking) {
            log(frameIndex, "[WARN] run(): there already exist tracking object.");
            mIsTracking = false;
        }

        if(!doObjectDetection) return null;

        // Step 2. Tracking 실패시 Yolo 기반으로 objects detection 을 한다
        // confidence 기준으로 큰 순으로 정렬하여 반환한다.
        List<Rect2d> targetBoxes = detectObjects(inputFrame, frameIndex);

        // Step 3. initialize object tracker
        if (targetBoxes.size() > 0) {

            mTracker = TrackerCSRT.create();

            Rect2d targetLocation = targetBoxes.get(0);

            for (Rect2d target_candidate_box : targetBoxes) {
                // x, y, w, h = target_candidate_box
                // target_candidate_box = (x, y, w, h)
                boolean tracker_initialized = mTracker.init(inputFrame, target_candidate_box);
                if (tracker_initialized) {
                    mIsTracking = true;
                    mTrackingObjectLocation = target_candidate_box;
                    mTrackingFailCount = 0;
                    targetLocation = target_candidate_box;
                    log(frameIndex, "run() tracker initialized, target_location=" + target_candidate_box);
                    break;
                }
            }

            // 가로/세로 비율대로 좌표 조정
            Rect adjTargetLocation = adjustRatio(new Size(width, height), targetLocation);

            // Resize 전 원래 이미지의 크기에 맞는 좌표로 변환
            return new Rect(
                    (int)(adjTargetLocation.x / mResizeFx),
                    (int)(adjTargetLocation.y / mResizeFy),
                    (int)(adjTargetLocation.width / mResizeFx),
                    (int)(adjTargetLocation.height / mResizeFy));
        }

        // TODO: EXPERIMENTAL - Find object location with margin if failed to detect above.
        // Step 4. Object detection 이 실패하면 마지막으로 인식된 object 위치에 margin 을 추가하여 반환한다
        if (HISTORICAL_BOUNDARY && mTrackingObjectLocation != null) {

            mTrackingFailCount += 1;

            if (mTrackingFailCount < HISTORICAL_BOUNDARY_MAX_TRY) {

                // 30번 연속 실패마다 margin 을 1% 씩 증가시킴
                int marginRatio = (int)(mTrackingFailCount / 30) + 1;

                double old_x = mTrackingObjectLocation.x, old_y = mTrackingObjectLocation.y;
                double old_w = mTrackingObjectLocation.width, old_h = mTrackingObjectLocation.height;

                // Margin 을 반영하여 새로운 x, y를 계산함
                double margin_w = old_w * marginRatio / 100;
                double margin_h = old_h * marginRatio / 100;

                double new_x = (old_x - margin_w >= 0) ? (old_x - margin_w) : 0;
                double new_w = (old_x + old_w + margin_w <= width) ? (old_w + margin_w) : (width - old_x);
                double new_y = (old_y - margin_h >= 0) ? (old_y - margin_h) : 0;
                double new_h = (old_y + old_h + margin_h <= height) ? (old_h + margin_h) : (height - old_y);

                // 가로/세로 비율대로 좌표 조정
                Rect adjTargetLocation = adjustRatio(new Size(width, height), new Rect2d(new_x, new_y, new_w, new_h));

                // Resize 전 원래 이미지의 크기에 맞는 좌표로 변환
                return new Rect(
                        (int)(adjTargetLocation.x / mResizeFx),
                        (int)(adjTargetLocation.y / mResizeFy),
                        (int)(adjTargetLocation.width / mResizeFx),
                        (int)(adjTargetLocation.height / mResizeFy));
            }
        }

        log(frameIndex, "run(): failed to detect target object");
        return null;
    }

    private Rect trackObject(Mat frame, int frameIndex) {

        String fail_message = "trackObject(): object location tracking failed, ";

        if (mTracker == null) {
            fail_message += "mTracker=null";
            log(frameIndex, fail_message);
            return null;
        }

        Rect2d newObjectLocation = new Rect2d();
        boolean tracked = mTracker.update(frame, newObjectLocation);

        // Tracking 실패
        if (!tracked) {
            fail_message += "tracked=false";
            log(frameIndex, fail_message);
            return null;
        }

        log(frameIndex, "trackObject(): found object location via tracking, location=" + newObjectLocation);

        double old_x = mTrackingObjectLocation.x, old_y = mTrackingObjectLocation.y;
        double old_w = mTrackingObjectLocation.width, old_h = mTrackingObjectLocation.height;
        double new_x = newObjectLocation.x, new_y = newObjectLocation.y;
        double new_w = newObjectLocation.width, new_h = newObjectLocation.height;

        // Check Area Change
        float old_area = (float) (old_w * old_h);
        float new_area = (float) (new_w * new_h);
        float area_changed = new_area / old_area;

        if (area_changed < (1 - AREA_CHANGED_THRESHOLD)) {
            fail_message += ("area_changed=" + area_changed);
            log(frameIndex, fail_message);
            return null;
        }

        if (area_changed > (1 + AREA_CHANGED_THRESHOLD)) {
            fail_message += ("area_changed=" + area_changed);
            log(frameIndex, fail_message);
            return null;
        }

        // Check Ratio Change
        float old_ratio = (float) (old_w / old_h);
        float new_ratio = (float) (new_w / new_h);
        float ratio_changed = new_ratio / old_ratio;

        if (ratio_changed < (1 - RATIO_CHANGED_THRESHOLD)) {
            fail_message += ("ratio_changed=" + ratio_changed);
            log(frameIndex, fail_message);
            return null;
        }

        if (ratio_changed > (1 + RATIO_CHANGED_THRESHOLD)) {
            fail_message += ("ratio_changed=" + ratio_changed);
            log(frameIndex, fail_message);
            return null;
        }

        // Check Ratio Threshold
        if (new_ratio < WH_RATIO_THRESHOLD) {
            fail_message +=  ("new_ratio=" + new_ratio);
            log(frameIndex, fail_message);
            return null;
        }

        // Check Movement Threshold
        float old_center_x = (float)(old_x + old_w / 2), old_center_y = (float)(old_y + old_h / 2);
        float new_center_x = (float)(new_x + new_w / 2), new_center_y = (float)(new_y + new_h / 2);

        float x_changed = Math.abs(new_center_x - old_center_x);
        float y_changed = Math.abs(new_center_y - old_center_y);

        int height = frame.height(), width = frame.width();

        if (x_changed > (width * MOVE_THRESHOLD) || y_changed > (height * MOVE_THRESHOLD)) {
            fail_message += ("x_changed=" + x_changed + ", y_changed=" + y_changed);
            log(frameIndex, fail_message);
            return null;
        }

        // Update location of object
        mTrackingObjectLocation = newObjectLocation;
        return adjustRatio(new Size(width, height), newObjectLocation);
    }

    private List<Rect2d> detectObjects(Mat frame, int frameIndex) {

        Mat rgbFrame = new Mat();
        //Imgproc.cvtColor(frame, rgbFrame, Imgproc.COLOR_BGRA2BGR);
        Mat blob = Dnn.blobFromImage(
                frame,1/255.0, new Size(INPUT_WIDTH, INPUT_HEIGHT),
                new Scalar(0,0,0),true,false);

        mNet.setInput(blob);
        List<Mat> outs = new ArrayList<>();
        mNet.forward(outs, mOutputLayers);

        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> objConfidences = new ArrayList<>();

        outs.forEach(out -> IntStream.range(0, out.rows()).forEach(detectionIndex -> {
            Mat detection = out.row(detectionIndex);
            Mat scores = detection.colRange(5, out.cols());
            Core.MinMaxLocResult r = Core.minMaxLoc(scores);

            int classId = (int) r.maxLoc.x;
            float confidence = (float) r.maxVal;

            if (confidence == 0.0f) return;

            if (confidence > CONFIDENCE_THRESHOLD)
            {
                float[] data = new float[1];
                Mat boundBox = detection.colRange(0, 5);

                boundBox.get(0, 0, data);
                // int centerX = (int)(data[0] * frame.cols());
                float centerX = data[0] * frame.cols();

                boundBox.get(0, 1, data);
                // int centerY = (int)(data[0] * frame.rows());
                float centerY = data[0] * frame.rows();

                boundBox.get(0, 2, data);
                // int width = (int)(data[0] * frame.cols() * RESIDUAL_BOUNDARY_RATIO);
                float width = data[0] * frame.cols() * RESIDUAL_BOUNDARY_RATIO;

                boundBox.get(0, 3, data);
                // int height = (int)(data[0] * frame.rows() * RESIDUAL_BOUNDARY_RATIO);
                float height = data[0] * frame.rows() * RESIDUAL_BOUNDARY_RATIO;

                // int left = centerX - width / 2;
                // int top = centerY - height / 2;
                float left = centerX - width / 2;
                float top = centerY - height / 2;

                boxes.add(new Rect2d(left, top, width, height));
                confidences.add(confidence);
                classIds.add(classId);
                boundBox.get(0, 4, data);
                objConfidences.add(data[0]);

            } else {
                log(frameIndex,"detectObjects(): " + mClasses.get(classId) + " detected," +
                        " but confidence is lower than threshold, confidence =" + confidence);
            }
        }));

        return detectTargetObject(classIds, confidences, boxes, objConfidences, frameIndex);
    }

    private List<Rect2d> detectTargetObject(List<Integer> classIds, List<Float> confidences,
                                  List<Rect2d> boxes, List<Float> objConfidences, int frameIndex) {

        List<Integer> targetObjectBoxesIndexes = new ArrayList<>();

        // MatOfRect boxesMat = new MatOfRect();
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(boxes);
        MatOfFloat configsMat = new MatOfFloat();
        configsMat.fromList(objConfidences);
        MatOfInt idxsMat = new MatOfInt();

        Dnn.NMSBoxes(boxesMat, configsMat, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD, idxsMat);

        List<Integer> nmsBoxesIdxs = idxsMat.toList();

        IntStream.range(0, boxes.size()).forEach( index -> {
            if (!nmsBoxesIdxs.contains(index)) return;

            String label = mClasses.get(classIds.get(index));
            if (!TRACKING_OBJECT_LABELS.contains(label)) {
                logv(frameIndex, "detectTargetObject(): ignore this object (label)," +
                        " label=" + label + ", box=" + boxes.get(index));
                return;
            }

            Rect2d box = boxes.get(index);
            float ratio = ((float) box.width) / ((float) box.height);
            if (ratio >= WH_RATIO_THRESHOLD) {
                targetObjectBoxesIndexes.add(index);
            }
            else {
                logv(frameIndex, "detectTargetObject(): ignore this object (ratio)," +
                        " label=" + label + ", box=" + boxes.get(index) + ", ratio=" + ratio);
            }
        });

        targetObjectBoxesIndexes.sort(Comparator.comparing(confidences::get));

        /*
        targetObjectBoxesIndexes.sort((index1, index2) ->
                Float.compare(confidences.get(index2), confidences.get(index1)));
         */

        Stream<Rect2d> targetObjectBoxesStream = targetObjectBoxesIndexes.stream().map(boxes::get);
        Stream<Float> targetObjectConfidencesStream = targetObjectBoxesIndexes.stream().map(confidences::get);

        log(frameIndex,"detectTargetObject(): target_object_boxes=" +
                Arrays.toString(targetObjectBoxesStream.toArray()) +
                ", target_object_confidences=" +
                Arrays.toString(targetObjectConfidencesStream.toArray()));

        targetObjectBoxesStream = targetObjectBoxesIndexes.stream().map(boxes::get);

        return targetObjectBoxesStream.collect(Collectors.toList());
    }

    @SuppressWarnings("unused")
    private void log(String message) {
        Log.d(LOG_TAG, message);
    }

    @SuppressWarnings("unused")
    private void log(int frameIndex, String message) {
        Log.d(LOG_TAG,"[" + mFrameIndexDf.format(frameIndex) + "] " + message);
    }

    @SuppressWarnings("unused")
    private void logv(String message) {
        Log.v(LOG_TAG, message);
    }

    @SuppressWarnings("unused")
    private void logv(int frameIndex, String message) {
        DecimalFormat df = new DecimalFormat("000000");
        Log.v(LOG_TAG,"[" + df.format(frameIndex) + "] " + message);
    }

    private Rect adjustRatio(Size size, Rect2d inputBoundBox) {

        double width = size.width, height = size.height;
        double x = inputBoundBox.x, y = inputBoundBox.y,
                w = inputBoundBox.width, h = inputBoundBox.height;

        double ratio = size.width / size.height;
        // double c_ratio = ((double) inputBoundBox.width) / ((double) inputBoundBox.height);
        double c_ratio = inputBoundBox.width / inputBoundBox.height;

        if (c_ratio < ratio) {

            double w_diff = (ratio * h - w) / 2;

            if (ratio * h >= width) {
                return toRect(0, 0, width, height);
            }

            if (x >= w_diff) {
                if ((x + w) +w_diff <= width) {
                    return toRect(x - w_diff, y, w + w_diff, h);
                }
                else {
                    double temp_w = width - (w + x); // 오른쪽에서 확장 가능한 크기

                    if (x >= w_diff + (w_diff - temp_w)) {
                        return toRect(x - w_diff - (w_diff - temp_w), y, w + temp_w, h);
                    }
                    else {
                        return toRect(0, 0, width, height);
                    }
                }
            }
            else {
                if (w + w_diff + w_diff <= width) {
                    return toRect(0, y, w + w_diff + w_diff, h);
                }
                else {
                    return toRect(0, 0, width, height);
                }
            }
        }
        else if (c_ratio > ratio) {

            double h_diff = (w / ratio - h) / 2;

            if (w / ratio >= width) {
                return toRect(0, 0, width, height);
            }

            if (y > h_diff) {
                if ((y + h) + h_diff <= height) {
                    return toRect(x, y - h_diff, w, h + h_diff);
                }
                else {
                    double temp_h = height - (h + y);

                    if (h >= h_diff + (h_diff - temp_h)) {
                        return toRect(x, y - h_diff - (h_diff - temp_h), w, h + temp_h);
                    }
                    else {
                        return toRect(0, 0, width, height);
                    }
                }
            }
            else {
                if (h + h_diff + h_diff <= height) {
                    return toRect(x, 0, w, h + h_diff + h_diff);
                }
                else {
                    return toRect(0, 0, width, height);
                }
            }
        } else {
            return toRect(x, y, w, h);
        }
    }

    private Rect toRect(double x, double y, double w, double h) {
        return new Rect(new double[]{x, y, w, h});
    }
}

