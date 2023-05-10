package com.lg2.beney.augmented_paper_detector.core;

import android.util.Log;

import androidx.core.util.Pair;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.stream.Collectors;

@SuppressWarnings("unused")
public class EdgeDetector {

    private final static String LOG_TAG = EdgeDetector.class.getSimpleName();
    private final static boolean VDBG = false;

    // Canny Features
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean USE_BILATERAL_FILTER = false;
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean USE_THRESHOLD = false;
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean USE_BLUR = false;
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean USE_MORPH_CLOSE = false;
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean USE_SCHARR = true;

    @SuppressWarnings("FieldCanBeLocal")
    private final int CANNY_THRESHOLD1 = 150;
    @SuppressWarnings("FieldCanBeLocal")
    private final int CANNY_THRESHOLD2 = 300;

    // Convex Hull Features
    private final int CONVEX_HULL_AREA_THRESHOLD = 800;

    // Save Intermediate steps.
    private final boolean SAVE_INTERMEDIATE_IMAGES = false;
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean SAVE_ORIGINAL = (SAVE_INTERMEDIATE_IMAGES || false);
    @SuppressWarnings("FieldCanBeLocal")
    private final int SAVE_ORIGINAL_STEP = 0;
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean SAVE_CANNY = (SAVE_INTERMEDIATE_IMAGES || false);
    @SuppressWarnings("FieldCanBeLocal")
    private final int SAVE_CANNY_STEP = 2;
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean SAVE_CONTOURS = (SAVE_INTERMEDIATE_IMAGES || false);
    @SuppressWarnings("FieldCanBeLocal")
    private final int SAVE_CONTOURS_STEP = 3;
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean SAVE_HULLS = (SAVE_INTERMEDIATE_IMAGES || false);
    @SuppressWarnings("FieldCanBeLocal")
    private final int SAVE_HULLS_STEP = 4;
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean SAVE_MERGE_HULLS = (SAVE_INTERMEDIATE_IMAGES || false);
    @SuppressWarnings("FieldCanBeLocal")
    private final int SAVE_MERGE_HULLS_STEP = 5;

    // Object Accept Function Constants
    @SuppressWarnings("FieldCanBeLocal")
    private final float ACCEPTABLE_OBJECT_WH_RATIO = 1.77f;
    @SuppressWarnings("FieldCanBeLocal")
    private final float ACCEPTABLE_OBJECT_WH_RATIO_MARGIN = 0.1f;
    @SuppressWarnings("FieldCanBeLocal")
    private final float ACCEPTABLE_OBJECT_AREA_RATIO = 0.7f;
    @SuppressWarnings("FieldCanBeLocal")
    private final float ACCEPTABLE_OBJECT_CENTER_MARGIN = 0.33333f;

    // Draw Failure Features
    private final boolean DRAW_FAILURES = false;
    //
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean DRAW_DROPPED_CONVEX_HULLS = (DRAW_FAILURES || false);
    private final Scalar DRAW_DROPPED_CONVEX_HULLS_COLOR = new Scalar(0, 255, 0); // BGR => Green
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean DRAW_ACCEPTED_CONVEX_HULLS = false;
    private final Scalar DRAW_ACCEPTED_CONVEX_HULLS_COLOR = new Scalar(255, 0, 0); // BGR => Blue;
    //
    @SuppressWarnings({"PointlessBooleanExpression", "ConstantConditions", "FieldCanBeLocal"})
    private final boolean DRAW_DROPPED_MERGED_CONVEX_HULLS = (DRAW_FAILURES || false);
    private final Scalar DRAW_DROPPED_MERGED_CONVEX_HULLS_COLOR = new Scalar(255, 255, 0);
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean DRAW_ACCEPTED_MERGED_CONVEX_HULLS = true;
    private final Scalar DRAW_ACCEPTED_MERGED_CONVEX_HULLS_COLOR = new Scalar(3, 106, 252);

    private final IntermediateRecorder mIntermediateRecorder;
    private final Random mRand;

    private final DecimalFormat mFrameIndexDf = new DecimalFormat("00000000");

    public EdgeDetector() {
        this(null);
    }

    public EdgeDetector(IntermediateRecorder intermediateRecorder) {
        mIntermediateRecorder = intermediateRecorder;
        mRand = new Random();
    }

    private Mat canny(Mat frame, int frameIndex, boolean saveImage) {

        // Detect Edges by MSER moment
        // return detect_edges(frame, remove_text=False)

        Mat bilateralFilterMat;
        if (USE_BILATERAL_FILTER) {
            bilateralFilterMat = new Mat();
            Imgproc.bilateralFilter(frame, bilateralFilterMat, 3, 100, 100);
        } else {
            bilateralFilterMat = frame;
        }

        Mat grayMat = new Mat();
        Imgproc.cvtColor(bilateralFilterMat, grayMat, Imgproc.COLOR_BGR2GRAY);

        Mat thresholdMat;
        if (USE_THRESHOLD) {
            thresholdMat = new Mat();
            Imgproc.threshold(grayMat, thresholdMat, 200, 255, Imgproc.THRESH_TRUNC);
            thresholdMat = thresholdMat.row(1);
        }
        else {
            thresholdMat = grayMat;
        }

        Mat blurredMat;
        if (USE_BLUR) {
            blurredMat = new Mat();
            // reduce noise with a 3x3 kernel
            // blurred_mat = cv2.GaussianBlur(threshold_mat, (3, 3), 0)
            Imgproc.blur(thresholdMat, blurredMat, new Size(3, 3));
        }
        else {
            blurredMat = thresholdMat;
        }

        Mat dilatedMat;
        if (USE_MORPH_CLOSE) {
            dilatedMat = new Mat();
            Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            // dilated_mat = Imgproc.dilate(blurred_mat, rectKernel, iterations=2)
            Imgproc.morphologyEx(blurredMat, dilatedMat, Imgproc.MORPH_CLOSE, rectKernel, new Point(-1,-1),5);
        }
        else {
            dilatedMat = blurredMat;
        }

        Mat cannyMat = new Mat();
        if (USE_SCHARR) {
            Mat scharrX = new Mat();
            Mat scharrY = new Mat();
            Imgproc.Scharr(dilatedMat, scharrX, CvType.CV_16S, 1, 0);
            Imgproc.Scharr(dilatedMat, scharrY, CvType.CV_16S, 0, 1);
            Imgproc.Canny(scharrX, scharrY, cannyMat, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
        }
        else {
            Imgproc.Canny(dilatedMat, cannyMat, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
        }

        //noinspection ConstantConditions
        if (SAVE_CANNY || saveImage) {
            if (mIntermediateRecorder != null) {
                mIntermediateRecorder.saveImage(cannyMat, frameIndex, SAVE_CANNY_STEP, "canny");
            }
        }

        return cannyMat;
    }

    public Pair<Mat, List<MatOfPoint>> contours(Mat frame, int frameIndex) {
        return this.contours(frame, frameIndex, false);
    }

    public Pair<Mat, List<MatOfPoint>> contours(Mat frame, int frameIndex, boolean saveImage) {

        //noinspection ConstantConditions
        if (SAVE_ORIGINAL || saveImage) {
            if (mIntermediateRecorder != null) {
                mIntermediateRecorder.saveImage(frame, frameIndex, SAVE_ORIGINAL_STEP, "original");
            }
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Mat cannyMat = canny(frame, frameIndex, saveImage);
        Imgproc.findContours(cannyMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_KCOS);

        //noinspection ConstantConditions
        if (SAVE_CONTOURS || saveImage) {
            if (mIntermediateRecorder != null) {
                Mat copied_frame = frame.clone();
                contours.forEach(contour -> {
                    Scalar c = new Scalar(mRand.nextInt(256), 0, mRand.nextInt(256));
                    Imgproc.drawContours(copied_frame, Collections.singletonList(contour), -1, c, 3);
                });
                mIntermediateRecorder.saveImage(copied_frame, frameIndex, SAVE_CONTOURS_STEP, "contours");
            }
        }

        return new Pair<>(cannyMat, contours);
    }

    public List<MatOfPoint> convexHulls(Mat frame, List<MatOfPoint> contours, int frameIndex) {
        return this.convexHulls(frame, contours, frameIndex, false);
    }

    public List<MatOfPoint> convexHulls(Mat frame, List<MatOfPoint> contours, int frameIndex, boolean saveImage) {

        List<MatOfPoint> hulls = new ArrayList<>();

        List<MatOfPoint> approxContours = contours.stream().map(contour -> {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double perimeter = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approximatedShape = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approximatedShape, 0.02 * perimeter, true);
            MatOfPoint approxPoint = new MatOfPoint();
            contour2f.convertTo(approxPoint, CvType.CV_32S);
            return approxPoint;
        }).collect(Collectors.toList());


        approxContours.forEach(contour -> {
            List<Point> allPoints = contour.toList();
            MatOfInt hullIdxs = new MatOfInt();
            Imgproc.convexHull(contour, hullIdxs, true);

            // Same as below line
            /*
            List<Point> convexPoints = hull.toList().stream().map(allPoints::get).collect(Collectors.toList());
            hulls.add(new MatOfPoint(convexPoints.toArray(new Point[0])));
            */

            MatOfPoint hull = new MatOfPoint(hullIdxs.toList().stream().map(allPoints::get).toArray(Point[]::new));
            if (Imgproc.contourArea(hull) >= CONVEX_HULL_AREA_THRESHOLD) {
                hulls.add(hull);
            }
        });


        //noinspection ConstantConditions
        if (SAVE_HULLS || saveImage) {
            if (mIntermediateRecorder != null) {
                Mat copiedFrame = frame.clone();
                hulls.forEach(hull -> {
                    Scalar c = new Scalar(mRand.nextInt(256), 0, mRand.nextInt(256));
                    Imgproc.drawContours(copiedFrame, Collections.singletonList(hull), -1, c, 3);
                });
                mIntermediateRecorder.saveImage(copiedFrame, frameIndex, SAVE_HULLS_STEP, "convex-hulls");
            }
        }

        return hulls;
    }

    // Return object candidate rectangles by merging hulls
    // Check acceptable: is_is_acceptable
    // Sort by area of rectangles: reverse order
    public List<Rect> mergeHulls(Mat frame, List<MatOfPoint> hulls, Rect objectBoundary, int frameIndex, boolean saveImage) {

        return this.mergeHulls(frame, hulls, new Rect2d(objectBoundary.tl(), objectBoundary.br()), frameIndex, saveImage, 5);
    }

    public List<Rect> mergeHulls(Mat frame, List<MatOfPoint> hulls, Rect2d objectBoundary, int frameIndex, boolean saveImage) {
        return this.mergeHulls(frame, hulls, objectBoundary, frameIndex, saveImage, 5);
    }

    public List<Rect> mergeHulls(Mat frame, List<MatOfPoint> hulls, Rect objectBoundary, int frameIndex, boolean saveImage, int max_num_results) {
        return this.mergeHulls(frame, hulls, new Rect2d(objectBoundary.tl(), objectBoundary.br()), frameIndex, saveImage, max_num_results);
    }

    public List<Rect> mergeHulls(Mat frame, List<MatOfPoint> hulls, Rect2d objectBoundary, int frameIndex, boolean saveImage, int max_num_results) {

        int height = frame.height(), width = frame.width();
        Point objectCenter = new Point();

        // Check object boundary and get center of the boundary
        if (objectBoundary == null) {
            objectBoundary = new Rect2d(0, 0, width, height);
            objectCenter.x = (double) width / 2;
            objectCenter.y = (double) height / 2;
        }
        else {
            objectCenter.x = objectBoundary.x + objectBoundary.width / 2;
            objectCenter.y = objectBoundary.y + objectBoundary.height / 2;
        }

        // Sort hulls by distance to object center
        hulls.sort(Comparator.comparing(hull -> distance(objectCenter, hull)));
        // hulls.sort(Comparator.comparingDouble(hull -> distance(objectCenter, hull)));

        List<MatOfPoint> stackedHulls = new ArrayList<>();
        List<Rect> acceptedRectangles = new ArrayList<>();

        // Object의 중심에 가장 가까운 순서대로 ConvexHull 을 조회함
        int index = 0;
        for (MatOfPoint hull : hulls) {

            // Object Detection 에 의해서 찾아진 boundary를 벗어나는 ConvexHull은 버림
            if (!isIn(hull, objectBoundary)) {
                //noinspection ConstantConditions
                if (DRAW_DROPPED_CONVEX_HULLS) {
                    Imgproc.drawContours(frame, Collections.singletonList(hull), -1, DRAW_DROPPED_CONVEX_HULLS_COLOR, 2);
                    continue;
                }
            }

            if (DRAW_ACCEPTED_CONVEX_HULLS) {
                Imgproc.drawContours(frame, Collections.singletonList(hull), -1, DRAW_ACCEPTED_CONVEX_HULLS_COLOR, 2);
            }

            // Stack Acceptable Hull
            stackedHulls.add(hull);
            MatOfPoint mergedHull = mergeHulls(stackedHulls);
            Rect mergedBoundingRect = Imgproc.boundingRect(mergedHull);

            // if the bounding_rect is not changed with adding the last hull, ignore it.
            if (acceptedRectangles.contains(mergedBoundingRect)) {
                continue;
            }

            boolean updated = false;
            if (isAcceptable(mergedBoundingRect, new Size (width, height))) {
                acceptedRectangles.add(mergedBoundingRect);
                if (DRAW_ACCEPTED_MERGED_CONVEX_HULLS) {
                    Imgproc.rectangle(
                            frame, mergedBoundingRect, DRAW_ACCEPTED_MERGED_CONVEX_HULLS_COLOR, 3);
                    updated = true;
                }
            }
            else {
                //noinspection ConstantConditions
                if (DRAW_DROPPED_MERGED_CONVEX_HULLS) {
                    Imgproc.rectangle(
                            frame, mergedBoundingRect, DRAW_DROPPED_MERGED_CONVEX_HULLS_COLOR, 3);
                    updated = true;
                }
            }

            // if any convex is allowed to draw in frame
            @SuppressWarnings("ConstantConditions")
            boolean drawHulls = DRAW_DROPPED_CONVEX_HULLS || DRAW_ACCEPTED_CONVEX_HULLS ||
                    DRAW_DROPPED_MERGED_CONVEX_HULLS || DRAW_ACCEPTED_MERGED_CONVEX_HULLS;

            //noinspection ConstantConditions
            if (drawHulls && updated && (SAVE_MERGE_HULLS || saveImage)){
                if (mIntermediateRecorder != null) {
                    mIntermediateRecorder.saveImage(frame, frameIndex, SAVE_MERGE_HULLS_STEP, index, "merge_hulls");
                }
            }

            index++;
        }

        // Sort by area and peek max {max_num_results} items
        acceptedRectangles.sort((Comparator.comparing(Rect::area).reversed()));
        // acceptedRectangles.sort((Comparator.comparingDouble(Rect::area).reversed()));

        return acceptedRectangles.size() > max_num_results ? acceptedRectangles.subList(0, max_num_results) : acceptedRectangles;
    }

    private double distance(Point point, MatOfPoint hull) {

        double hullX = point.x, hullY = point.y;
        Moments moments = Imgproc.moments(hull);

        if (moments.get_m00() != 0) {
            hullX = moments.get_m10() / moments.get_m00();
            hullY = moments.get_m01() / moments.get_m00();
        }

        return Math.hypot(point.x - hullX, point.y - hullY);
    }

    private double distance2(Point point, MatOfPoint hull) {

        List<Point> points = hull.toList();
        OptionalDouble distance2center = points.stream().
                map(hPoint -> Math.hypot(hPoint.x - point.x, hPoint.y - point.y)).
                mapToDouble(x->x).average();

        return distance2center.isPresent() ? distance2center.getAsDouble() : Double.MAX_VALUE;
    }

    private boolean isIn(MatOfPoint hull, Rect2d boundary) {

        Rect hullBoundRect = Imgproc.boundingRect(hull);

        if (boundary == null) {
            return true;
        }

        double hullX = hullBoundRect.x, hullY = hullBoundRect.y;
        double hullW = hullBoundRect.width, hullH = hullBoundRect.height;

        double boundaryX = boundary.x, boundaryY = boundary.y;
        double boundaryW = boundary.width, boundaryH = boundary.height;

        if ((hullX >= boundaryX) && ((hullX + hullW) <= (boundaryX + boundaryW))) {
            return (hullY >= boundaryY) && ((hullY + hullH) <= (boundaryY + boundaryH));
        }

        return false;
    }

    private boolean isAcceptable(Rect objectRect, Size frameSize) {

        double x = (double) objectRect.x, y = (double) objectRect.y;
        double w = (double) objectRect.width, h = (double) objectRect.height;
        double objectWhRatioMargin = (w / h) - ACCEPTABLE_OBJECT_WH_RATIO;
        double width = frameSize.width, height = frameSize.height;
        double objectAreaRatio = (w * h) / (width * height);

        logv("isAcceptable(): objectRect=" + objectRect + ", frameSize=" + frameSize);

        if ((-ACCEPTABLE_OBJECT_WH_RATIO_MARGIN <= objectWhRatioMargin) &&
                (objectWhRatioMargin <= ACCEPTABLE_OBJECT_WH_RATIO_MARGIN)) {
            if (objectAreaRatio <= ACCEPTABLE_OBJECT_AREA_RATIO) {

                double center_x = x + w / 2;
                double center_y = y + h / 2;

                double widthMargin = width * ACCEPTABLE_OBJECT_CENTER_MARGIN;
                double heightMargin = height * ACCEPTABLE_OBJECT_CENTER_MARGIN;

                return (widthMargin <= center_x) && (center_x <= (width - widthMargin)) &&
                        (heightMargin <= center_y) && (center_y <= (height - heightMargin));
            } else {
                logv("isAcceptable(): Not acceptable, objectAreaRatio=" + objectAreaRatio);
            }
        }
        else {
            logv("isAcceptable(): Not acceptable, objectWhRatioMargin=" + objectWhRatioMargin);
        }

        return false;
    }

    private MatOfPoint mergeHulls(List<MatOfPoint> hulls) {

        // Find ConvexHull including all points in CROP IMAGE
        List<Point> hullPoints = new ArrayList<>();
        hulls.forEach(hull -> hullPoints.addAll(hull.toList()));

        MatOfInt convexHullIdxs = new MatOfInt();
        Imgproc.convexHull(new MatOfPoint(hullPoints.toArray(new Point[0])), convexHullIdxs, true);
        return new MatOfPoint(convexHullIdxs.toList().stream().map(hullPoints::get).toArray(Point[]::new));
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
        if (VDBG) {
            Log.v(LOG_TAG, message);
        }
    }

    @SuppressWarnings("unused")
    private void logv(int frameIndex, String message) {
        if (VDBG) {
            DecimalFormat df = new DecimalFormat("000000");
            Log.v(LOG_TAG, "[" + df.format(frameIndex) + "] " + message);
        }
    }
}

