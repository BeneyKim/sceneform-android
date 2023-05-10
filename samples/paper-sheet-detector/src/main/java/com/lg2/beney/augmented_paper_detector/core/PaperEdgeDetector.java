package com.lg2.beney.augmented_paper_detector.core;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.util.Pair;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Triple;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
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
import java.util.stream.IntStream;

@SuppressWarnings("unused")
public class PaperEdgeDetector {

    private final static String LOG_TAG = PaperEdgeDetector.class.getSimpleName();
    private final static boolean VDBG = false;

    // Canny Features
    private static final boolean USE_BILATERAL_FILTER = false;
    private static final int BILATERAL_SIGMA_COLOR = 50;
    private static final int BILATERAL_SIGMA_SPACE = 3;

    private static final boolean USE_HSV_FILTER = true;
    private static final int HSV_SENSITIVITY = 90;

    private static final boolean USE_THRESHOLD = false;
    private static final boolean USE_BLUR = false;
    private static final boolean USE_MORPH_CLOSE = false;
    private static final boolean USE_SCHARR = true;

    private static final boolean USE_HOUGH = true;
    private static final double HOUGH_RESIZE_SCALE = 0.3f;
    private static final double HOUGH_INVERSE_RESIZE_SCALE = 1/0.3f;
    private static final boolean USE_HOUGH_SPLIT_HV = true;
    private static final boolean USE_HOUGH_FILTER = true;  // Under Checking
    private static final boolean USE_HOUGH_HISTORY = false; // Under Checking
    private static final int HOUGH_UNIT_ANGLES = 5; // 얼마나 기울어진 각도까지 Paper로 인식할지 기준
    private static final double HOUGH_UNIT_ANGLES_PI = Math.PI / (90.0 / HOUGH_UNIT_ANGLES);
    private static final double [][] HOUGH_UNIT_ANGLES_DROPS = {
            {HOUGH_UNIT_ANGLES_PI, Math.PI / 2 - HOUGH_UNIT_ANGLES_PI},
            {Math.PI / 2 + HOUGH_UNIT_ANGLES_PI, Math.PI - HOUGH_UNIT_ANGLES_PI}};
    private static final double [][] HOUGH_UNIT_ANGLES_VERTICAL = {
            {Math.PI / 2 - HOUGH_UNIT_ANGLES_PI, Math.PI / 2 + HOUGH_UNIT_ANGLES_PI}};

    private static final double [][] HOUGH_UNIT_ANGLES_HORIZONTAL = {
            {0, HOUGH_UNIT_ANGLES_PI},
            {Math.PI - HOUGH_UNIT_ANGLES_PI, Math.PI}};

    private static final boolean USE_AUTO_CANNY = true;
    // NORMAL IMAGE
    @SuppressWarnings("FieldCanBeLocal")
    private static final int CANNY_THRESHOLD1 = 150;
    @SuppressWarnings("FieldCanBeLocal")
    private static final int CANNY_THRESHOLD2 = 300;
    // DARK IMAGE
    // @SuppressWarnings("FieldCanBeLocal")
    // private final int CANNY_THRESHOLD1 = 50;
    // @SuppressWarnings("FieldCanBeLocal")
    // private final int CANNY_THRESHOLD2 = 150;

    // Convex Hull Features
    private static final int CONVEX_HULL_AREA_THRESHOLD = 800;
    private static final double CONVEX_HULL_AREA_RATIO_THRESHOLD = 0.001;
    private static final double CONVEX_HULL_IF_TOUCHED_MARGIN = 0.02;
    // Merge Hull Features
    private static final int MERGE_HULL_CANDIDATE_COUNT = 1;
    private int GAUSSIAN_KERNEL_SIZE = 9;

    // Object Accept Function Constants
    @SuppressWarnings("FieldCanBeLocal")
    private static final float ACCEPTABLE_OBJECT_WH_RATIO = 1.414f;
    @SuppressWarnings("FieldCanBeLocal")
    private static final float ACCEPTABLE_OBJECT_WH_RATIO_MARGIN = 0.1f;
    @SuppressWarnings("FieldCanBeLocal")
    private static final float ACCEPTABLE_OBJECT_AREA_RATIO = 0.9f;
    @SuppressWarnings("FieldCanBeLocal")
    private static final float ACCEPTABLE_OBJECT_CENTER_MARGIN = 0.33333f;

    private final Random mRand;
    private final boolean ED_ONLY;

    private final DecimalFormat mFrameIndexDf = new DecimalFormat("00000000");

    private List<Pair<Point, Point>> prevHvHoughLines;
    private List<Pair<Point, Point>> prevHorizontalHoughLines;
    private List<Pair<Point, Point>> prevVerticalHoughLines;


    public PaperEdgeDetector(boolean ed_only) {
        mRand = new Random();
        ED_ONLY = ed_only;
    }

    public PaperEdgeDetector(boolean ed_only, int kernel) {
        mRand = new Random();
        ED_ONLY = ed_only;
        GAUSSIAN_KERNEL_SIZE = kernel;
    }

    private Triple<Mat, Mat, Double> canny(Mat frame, Mat debugMat) {

        int height = frame.height(), width = frame.width();

        // ##### BILATERAL FILTER #####
        Mat bilateralMat;
        if (USE_BILATERAL_FILTER) {
            bilateralMat = new Mat();
            Imgproc.bilateralFilter(frame, bilateralMat, -1, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);
        } else {
            bilateralMat = frame;
        }

        // ##### HSV FILTER #####
        Mat hsvMat;
        if (USE_HSV_FILTER) {
            Mat hsvInput = new Mat();
            Imgproc.cvtColor(bilateralMat, hsvInput, Imgproc.COLOR_RGB2HSV);

            // int sensitivity = (255 - frame_center_pixel.mean()) * 0.8
            int sensitivity = HSV_SENSITIVITY;
            Scalar lowerWhite = new Scalar(0, 0, 255 - sensitivity);
            Scalar upperWhite = new Scalar(255, sensitivity, 255);

            // preparing the mask to overlay
            Mat mask = new Mat();
            Core.inRange(hsvInput, lowerWhite, upperWhite, mask);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11, 11));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);

            // The black region in the mask has the value of 0,
            // so when multiplied with original image removes all non - blue regions
            hsvMat = new Mat();
            Core.bitwise_and(frame, frame, hsvMat, mask);

            hsvInput.release();
            mask.release();
        } else {
            hsvMat = bilateralMat;
        }

        // ##### GRAY SCALE #####
        Mat grayMat = new Mat();
        Imgproc.cvtColor(hsvMat, grayMat, Imgproc.COLOR_BGR2GRAY);

        // CENTER by 5 division
        Mat whiteMeanROI = grayMat.submat((int)(height * 0.4), (int)(height * 0.6), (int)(width * 0.4), (int)(width * 0.6));
        byte[] data = new byte[whiteMeanROI.height() * whiteMeanROI.width() * whiteMeanROI.channels()];
        whiteMeanROI.get(0, 0, data);
        OptionalDouble average = IntStream.range(0, data.length).map(i -> data[i] & 0xFF).filter(x -> x != 0).average();
        double whiteMean = average.isPresent() ? average.getAsDouble() : 0.0f;

        // ##### THRESHOLD #####
        Mat thresholdMat;
        if (USE_THRESHOLD) {
            thresholdMat = new Mat();
            Imgproc.threshold(grayMat, thresholdMat,whiteMean * 0.90,255, Imgproc.THRESH_TOZERO);
        } else {
            thresholdMat = grayMat;
        }

        // ##### Gaussian BLUR #####
        Mat blurredMat;
        if (USE_BLUR) {
            blurredMat = new Mat();
            Imgproc.GaussianBlur(thresholdMat, blurredMat, new Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0, 0);
        } else {
            blurredMat = thresholdMat;
        }

        // ##### MORPH CLOSE #####
        // https://webnautes.tistory.com/1257
        Mat dilatedMat;
        if (USE_MORPH_CLOSE) {
            dilatedMat = new Mat();
            Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.morphologyEx(blurredMat, dilatedMat, Imgproc.MORPH_CLOSE, rectKernel, new Point(-1,-1),1);
        } else {
            dilatedMat = blurredMat;
        }

        // ##### Canny #####
        int cannyThreshold1 = CANNY_THRESHOLD1;
        int cannyThreshold2 = CANNY_THRESHOLD2;

        if (USE_AUTO_CANNY) {
            Scalar brightness = Core.mean(grayMat);
            double brightnessMean = brightness.val[0];
            double sigma = 0.33;
            cannyThreshold1 = (int)(Math.max(0, (1.0 - sigma) * brightnessMean));
            cannyThreshold2 = (int)(Math.min(255, (1.0 + sigma) * brightnessMean));
        }

        Mat cannyMat = new Mat();
        if (USE_SCHARR) {
            Mat scharrX = new Mat();
            Mat scharrY = new Mat();
            Imgproc.Scharr(dilatedMat, scharrX, CvType.CV_16S, 1, 0);
            Imgproc.Scharr(dilatedMat, scharrY, CvType.CV_16S, 0, 1);
            Imgproc.Canny(scharrX, scharrY, cannyMat, cannyThreshold1, cannyThreshold2);
            scharrX.release();
            scharrY.release();
        }
        else {
            Imgproc.Canny(dilatedMat, cannyMat, cannyThreshold1, cannyThreshold2);
        }

        // ##### HOUGH #####
        Mat houghMat;

        if (USE_HOUGH) {
            houghMat = Mat.zeros(new Size(cannyMat.width(), cannyMat.height()), cannyMat.type());

            if (USE_HOUGH_SPLIT_HV) {
                // 가로
                List<Pair<Point, Point>> horizontalLines = houghHorizontalLines(cannyMat, houghMat, 1, 1 * Math.PI / 180, 65);
                int hl_count = ((horizontalLines == null) ? 0 : horizontalLines.size());
                logv("canny(): " + hl_count  + " horizontal hough lines found.");
                // 세로
                List<Pair<Point, Point>> verticalLines = houghVerticalLines(cannyMat, houghMat, 1, 1 * Math.PI / 180, 75);
                int vl_count = ((verticalLines == null) ? 0 : verticalLines.size());
                logv("canny(): " + vl_count + " vertical hough lines found.");

                if (debugMat != null) {
                    drawHoughLines(debugMat, horizontalLines);
                    drawHoughLines(debugMat, verticalLines);
                }

            } else {
                List<Pair<Point, Point>> hvLines = houghLines(cannyMat, houghMat, 1, 1 * Math.PI / 180, 70);
                int hvl_count = ((hvLines == null) ? 0 : hvLines.size());
                logv("canny(): " + hvl_count + " hough lines found.");

                if (debugMat != null) {
                    drawHoughLines(debugMat, hvLines);
                }
            }
        }
        else {
            houghMat = cannyMat;
        }

        boolean UNDER_CANNY_TEST = false;
        if (UNDER_CANNY_TEST) {
            double alpha = 0.5;
            Mat output = new Mat();
            Imgproc.cvtColor(houghMat, houghMat, Imgproc.COLOR_GRAY2RGB);
            Core.addWeighted(frame, 1 - alpha, houghMat, alpha, 0, output);
            return Triple.of(frame, grayMat, whiteMean);
        }

        return Triple.of(houghMat, grayMat, whiteMean);
    }

    private List<Pair<Point, Point>> houghLines(Mat inImg, Mat outImg, double iRho, double iTheta, int threshold) {

        Mat inResizeMat = new Mat();
        Imgproc.resize(inImg, inResizeMat, new Size(0,0), HOUGH_RESIZE_SCALE, HOUGH_RESIZE_SCALE);

        int hvLineCount = 0;
        Mat lines = new Mat();
        Imgproc.HoughLines(inResizeMat, lines, iRho, iTheta, threshold);

        if (lines.rows() == 0) {
            logv("houghLines(): No HoughLines");
            lines.release();
            inResizeMat.release();
            return null;
        }

        if (USE_HOUGH_FILTER) {
            lines = filterHoughLines(lines);
        }

        List<Pair<Point, Point>> hvLines = new ArrayList<>();

        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];

            boolean ignore = false;

            for (double[] houghUnitAnglesDrop : HOUGH_UNIT_ANGLES_DROPS) {
                double start = houghUnitAnglesDrop[0];
                double end = houghUnitAnglesDrop[1];
                if ((start < theta) && (theta < end)) {
                    ignore = true;
                    break;
                }
            }

            if (ignore) continue;

            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));

            pt1.x = pt1.x * HOUGH_INVERSE_RESIZE_SCALE;
            pt1.y = pt1.y * HOUGH_INVERSE_RESIZE_SCALE;
            pt2.x = pt2.x * HOUGH_INVERSE_RESIZE_SCALE;
            pt2.y = pt2.y * HOUGH_INVERSE_RESIZE_SCALE;

            hvLines.add(new Pair<>(pt1, pt2));
            hvLineCount += 1;
        }

        drawHoughLines(outImg, hvLines);

        if (USE_HOUGH_HISTORY) {
            drawHoughLines(outImg, this.prevHvHoughLines);
        }

        this.prevHvHoughLines = hvLines;
        log("houghLines(): Reduced from " + lines.rows() + " to " + hvLineCount);

        lines.release();
        inResizeMat.release();
        return hvLines;
    }

    private List<Pair<Point, Point>> houghHorizontalLines(Mat inImg, Mat outImg, double iRho, double iTheta, int threshold) {

        Mat inResizeMat = new Mat();
        Imgproc.resize(inImg, inResizeMat, new Size(0,0), HOUGH_RESIZE_SCALE, HOUGH_RESIZE_SCALE);

        int horizontalLineCount = 0;
        Mat lines = new Mat();
        Imgproc.HoughLines(inResizeMat, lines, iRho, iTheta, threshold);

        if (lines.rows() == 0) {
            log("houghHorizontalLines(): No HoughLines");
            lines.release();
            inResizeMat.release();
            return null;
        }

        if (USE_HOUGH_FILTER) {
            lines = filterHoughLines(lines);
        }

        List<Pair<Point, Point>> hLines = new ArrayList<>();

        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];

            for (double[] houghUnitAngles : HOUGH_UNIT_ANGLES_HORIZONTAL) {
                double start = houghUnitAngles[0];
                double end = houghUnitAngles[1];
                if ((start < theta) && (theta < end)) {

                    double a = Math.cos(theta), b = Math.sin(theta);
                    double x0 = a * rho, y0 = b * rho;
                    Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
                    Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));

                    pt1.x = pt1.x * HOUGH_INVERSE_RESIZE_SCALE;
                    pt1.y = pt1.y * HOUGH_INVERSE_RESIZE_SCALE;
                    pt2.x = pt2.x * HOUGH_INVERSE_RESIZE_SCALE;
                    pt2.y = pt2.y * HOUGH_INVERSE_RESIZE_SCALE;

                    hLines.add(new Pair<>(pt1, pt2));
                    horizontalLineCount += 1;
                }
            }
        }

        drawHoughLines(outImg, hLines);

        if (USE_HOUGH_HISTORY) {
            drawHoughLines(outImg, this.prevHorizontalHoughLines);
        }

        this.prevHorizontalHoughLines = hLines;
        logv("houghHorizontalLines(): Reduced from " + lines.rows() + " to " + horizontalLineCount);

        lines.release();
        inResizeMat.release();
        return hLines;
    }

    private List<Pair<Point, Point>> houghVerticalLines(Mat inImg, Mat outImg, double iRho, double iTheta, int threshold) {

        Mat inResizeMat = new Mat();
        Imgproc.resize(inImg, inResizeMat, new Size(0,0), HOUGH_RESIZE_SCALE, HOUGH_RESIZE_SCALE);

        int verticalLineCount = 0;
        Mat lines = new Mat();
        Imgproc.HoughLines(inResizeMat, lines, iRho, iTheta, threshold);

        if (lines.rows() == 0) {
            log("houghVerticalLines(): No HoughLines");
            lines.release();
            inResizeMat.release();
            return null;
        }

        if (USE_HOUGH_FILTER) {
            lines = filterHoughLines(lines);
        }

        List<Pair<Point, Point>> vLines = new ArrayList<>();

        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];

            for (double[] houghUnitAngles : HOUGH_UNIT_ANGLES_VERTICAL) {
                double start = houghUnitAngles[0];
                double end = houghUnitAngles[1];
                if ((start < theta) && (theta < end)) {

                    double a = Math.cos(theta), b = Math.sin(theta);
                    double x0 = a * rho, y0 = b * rho;
                    Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
                    Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));

                    pt1.x = pt1.x * HOUGH_INVERSE_RESIZE_SCALE;
                    pt1.y = pt1.y * HOUGH_INVERSE_RESIZE_SCALE;
                    pt2.x = pt2.x * HOUGH_INVERSE_RESIZE_SCALE;
                    pt2.y = pt2.y * HOUGH_INVERSE_RESIZE_SCALE;

                    vLines.add(new Pair<>(pt1, pt2));
                    verticalLineCount += 1;
                }
            }
        }

        drawHoughLines(outImg, vLines);

        if (USE_HOUGH_HISTORY) {
            drawHoughLines(outImg, this.prevVerticalHoughLines);
        }

        this.prevVerticalHoughLines = vLines;
        logv("houghVerticalLines(): Reduced from " + lines.rows() + " to " + verticalLineCount);

        lines.release();
        inResizeMat.release();

        return vLines;
    }

    private Mat filterHoughLines(Mat lines){
        return filterHoughLines(lines, lines.rows()); // Unique Lines
    }

    private Mat filterHoughLines(Mat lines, int num_lines_to_find) {

        if (lines.rows() < num_lines_to_find) {
            num_lines_to_find = lines.rows();
        }

        List<Integer> uniqueLineIndexes = new ArrayList<>();
        uniqueLineIndexes.add(0);

        List<Double> uniqueRhoes = new ArrayList<>();
        uniqueRhoes.add(lines.get(0, 0)[0]);

        List<Double> uniqueThetas = new ArrayList<>();
        uniqueThetas.add(lines.get(0, 0)[1]);

        // Filter the lines
        for (int x = 1; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 0)[1];

            // For this line, check which of the existing 4 it is similar to.
            boolean similar_rho = isSimilar(rho, uniqueRhoes, 10.0); // #10 pixels *
            boolean similar_theta = isSimilar(theta, uniqueThetas, Math.PI / 18.0);  // # 5 degrees
            boolean similar = (similar_rho && similar_theta);

            if (!similar){
                logv("filter_hough_lines(): Found a unique line: " + x + " rho = " + rho + " theta = " + theta);
                uniqueLineIndexes.add(x);
                uniqueRhoes.add(rho);
                uniqueThetas.add(theta);
            }

            if (uniqueLineIndexes.size() >= num_lines_to_find) {
                logv("filterHoughLines(): Found "+ num_lines_to_find + " unique lines!");
                break;
            }
        }

        Mat filteredLines = new Mat(uniqueLineIndexes.size(), lines.cols(), lines.type());
        for (int i=0; i<uniqueLineIndexes.size(); i++) {
            filteredLines.put(i, 0, lines.get(uniqueLineIndexes.get(i), 0));
        }

        log("filterHoughLines(): Reduced from " + lines.rows() + " to " + uniqueLineIndexes.size());

        return filteredLines;
    }

    private boolean isSimilar(double rho, List<Double> samples) {
        return isSimilar(rho, samples, 1e-08, 1e-05);
    }

    private boolean isSimilar(double rho, List<Double> samples, double atol) {
        return isSimilar(rho, samples, atol, 1e-05);
    }

    private boolean isSimilar(double rho, List<Double> samples, double atol, double rtol) {
        return samples.stream().anyMatch(x -> Math.abs(rho - x) <= (atol + rtol * Math.abs(x)));
    }

    private void drawHoughLines(Mat outImg, List<Pair<Point, Point>> lines) {

        if (outImg == null) return;
        if (lines == null || lines.size() == 0) return;

        for (Pair<Point, Point> line: lines) {
            Point pt1 = line.first;
            Point pt2 = line.second;

            if (outImg.channels() >= 3) {
                Imgproc.line(outImg, pt1, pt2, new Scalar(255, 255, 0), 3);
            } else {
                Imgproc.line(outImg, pt1, pt2, new Scalar(255), 3);
            }
        }
    }

    public List<MatOfPoint> getConvexHulls(Mat frame, Mat debugMat) {

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Triple<Mat, Mat, Double> cannyResult = canny(frame, debugMat);
        final Mat cannyMat = cannyResult.getLeft();
        final Mat grayMat = cannyResult.getMiddle();
        final double whiteMean = cannyResult.getRight();

        Imgproc.findContours(cannyMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_TC89_KCOS);

        final int height = frame.height(), width = frame.width();
        final double wMargin = width * CONVEX_HULL_IF_TOUCHED_MARGIN;
        final double hMargin = height * CONVEX_HULL_IF_TOUCHED_MARGIN;
        final Rect2d hullBoundary = new Rect2d(wMargin, hMargin, (width - 2 * wMargin), (height - 2 * hMargin));
        final double convexHullAreaThreshold = (CONVEX_HULL_AREA_RATIO_THRESHOLD * height * width);

        //noinspection DoubleNegation
        return contours.stream()
                .filter(contour -> !(!isIn(contour, hullBoundary, false) /* && drawDropContour(debugMat, contour, "B") */))
                .filter(contour -> !((Imgproc.contourArea(contour) < convexHullAreaThreshold) && drawDropContour(debugMat, contour, "A")))
                .filter(contour -> !((!isAcceptableColor(grayMat, hullCenter(contour), whiteMean)) && drawDropContour(debugMat, contour, "C")))
                .map(contour -> OpenCvUtils.convexHull(OpenCvUtils.approxPolyDP(contour, 0.01, true), true))
                // .filter(hull -> !(!(hull.rows() >= 4) && drawDropContour(debugMat, hull, "SS")))
                .filter(hull -> !(!(hull.rows() <= 6) && drawDropContour(debugMat, hull, "LS")))

                // Candidates
                .filter(hull -> {
                    Scalar c = new Scalar(mRand.nextInt(155) + 100, mRand.nextInt(200) + 50, mRand.nextInt(155) + 100);
                    Imgproc.drawContours(debugMat, Collections.singletonList(hull), -1, c, -1);
                    return true;
                })
                .collect(Collectors.toList());
    }

    private boolean isAcceptableColor(Mat grayMat, Point hullCenterPoint, double whiteMean) {
        final double whiteThreshold = whiteMean * 0.85;
        final int hx = (int) hullCenterPoint.x;
        final int hy = (int) hullCenterPoint.y;
        final double hcPixel = grayMat.get(hy, hx)[0];
        return hcPixel > whiteThreshold;
    }

    // Return object candidate rectangles by merging hulls
    // Check acceptable: is_is_acceptable
    // Sort by area of rectangles: reverse order
    public List<MatOfPoint> mergeHulls(Mat frame, Mat debugMat, List<MatOfPoint> hulls) {
        return this.mergeHulls(frame, debugMat, hulls, MERGE_HULL_CANDIDATE_COUNT);
    }

    public List<MatOfPoint> mergeHulls(Mat frame, Mat debugMat, List<MatOfPoint> hulls, int max_num_results) {

        final int height = frame.height(), width = frame.width();
        Point objectCenter = rectCenter(new Rect2d(0, 0, width, height));
        Imgproc.circle(debugMat, objectCenter, 5, new Scalar(0, 255, 0), -1);

        // Sort hulls by distance to object center
        hulls.sort(Comparator.comparing(hull -> distance(objectCenter, hull)));
        // hulls.sort(Comparator.comparingDouble(hull -> distance(objectCenter, hull)));

        final List<MatOfPoint> acceptedPolygons = new ArrayList<>();
        MatOfPoint lastMergedHull = null;

        // Object의 중심에 가장 가까운 순서대로 ConvexHull 을 조회함
        for (MatOfPoint hull : hulls) {

            MatOfPoint mergedHullCandidate = _mergeHulls(hull, lastMergedHull);
            if (lastMergedHull != null && !isAcceptableArea(hull, lastMergedHull, mergedHullCandidate)) {
                drawDropContour(debugMat, hull, "MA");
                continue;
            }

            Scalar c = new Scalar(mRand.nextInt(155) + 100, mRand.nextInt(200) + 50, mRand.nextInt(155) + 100);
            Imgproc.drawContours(debugMat, Collections.singletonList(hull), -1, c, -1);

            Pair<MatOfPoint, Double> approxPolyDPwAreaResult = OpenCvUtils.approxPolyDPwArea(mergedHullCandidate, 0.02, true);
            lastMergedHull = approxPolyDPwAreaResult.first;
            final double maa = Imgproc.contourArea(lastMergedHull);
            final double rra = approxPolyDPwAreaResult.second;

            // if the bounding_rect is not changed with adding the last hull, ignore it.
            if (acceptedPolygons.contains(lastMergedHull)) {
                continue;
            }

            if ((lastMergedHull.rows() == 4) && (maa / rra) >= 0.8) {
                acceptedPolygons.add(lastMergedHull);
                Imgproc.drawContours(debugMat, Collections.singletonList(lastMergedHull), -1, new Scalar(0, 0, 255), 20);
            }
        }

        // Sort by area and peek max {max_num_results} items
        acceptedPolygons.sort((Comparator.comparingDouble(x -> Imgproc.contourArea((MatOfPoint)x)).reversed()));
        // acceptedRectangles.sort((Comparator.comparingDouble(Rect::area).reversed()));
        // acceptedRectangles.sort(Comparator.comparing(item -> Math.abs(((float) item.width / (float)item.height) - ACCEPTABLE_OBJECT_WH_RATIO)));

        return acceptedPolygons.size() > max_num_results ? acceptedPolygons.subList(0, max_num_results) : acceptedPolygons;
    }

    private boolean isAcceptableArea(MatOfPoint newHull, MatOfPoint curMergedHull, MatOfPoint nxtMergedHull) {
        final double newHullArea = rotatedRectArea(newHull);
        final double curMergedHullArea = rotatedRectArea(curMergedHull);
        final double nxtMergedHullArea = rotatedRectArea(nxtMergedHull);
        final double residual_area_ratio = 1.0 - (curMergedHullArea + newHullArea) / nxtMergedHullArea;
        return residual_area_ratio < 0.15;
    }

    private static final Scalar DROP_COLOR = new Scalar(0, 0, 0);

    private boolean drawDropContour(Mat imgMat, MatOfPoint contour, String cause) {
        return drawDropContour(imgMat, contour, cause, DROP_COLOR);
    }
    @SuppressWarnings("SameParameterValue")
    private boolean drawDropContour(Mat imgMat, MatOfPoint contour, String cause, Scalar dropColor) {
        if (StringUtils.isEmpty(cause)) cause = "U";
        if (cause.length() > 2)
            cause = cause.substring(0, 2); // Draw the first two letters
        Imgproc.drawContours(imgMat, Collections.singletonList(contour), -1, dropColor, -1);
        Imgproc.putText(imgMat, cause, hullCenter(contour), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(200, 50, 50), 2);
        return true;
    }

    private double rotatedRectArea(MatOfPoint matOfPoint) {
        return Imgproc.minAreaRect(new MatOfPoint2f(matOfPoint.toArray())).size.area();
    }

    private Point hullCenter(MatOfPoint hull) {

        Moments moments = Imgproc.moments(hull);

        if (moments.get_m00() != 0) {
            double hullX = moments.get_m10() / moments.get_m00();
            double hullY = moments.get_m01() / moments.get_m00();
            return new Point(hullX, hullY);
        }

        return new Point(-1, -1);
    }

    private Point rectCenter(Rect2d rect) {
        return new Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    private double distance(Point point, MatOfPoint hull) {

        double hullX = point.x, hullY = point.y;
        Moments moments = Imgproc.moments(hull);

        if (moments.get_m00() != 0) {
            hullX = moments.get_m10() / moments.get_m00();
            hullY = moments.get_m01() / moments.get_m00();
        }

        double distance_x = point.x - hullX;
        double distance_y = point.y - hullY;

        distance_x /= ACCEPTABLE_OBJECT_WH_RATIO;
        return Math.hypot(distance_x, distance_y);
    }

    private boolean isIn(MatOfPoint hull, Rect2d boundary) {
        return isIn(hull, boundary, false);
    }

    @SuppressWarnings("SameParameterValue")
    private boolean isIn(MatOfPoint hull, Rect2d boundary, boolean allowBoundary) {

        Rect hullBoundRect = Imgproc.boundingRect(hull);

        if (boundary == null) {
            return true;
        }

        double hullX = hullBoundRect.x, hullY = hullBoundRect.y;
        double hullW = hullBoundRect.width, hullH = hullBoundRect.height;

        double boundaryX = boundary.x, boundaryY = boundary.y;
        double boundaryW = boundary.width, boundaryH = boundary.height;

        if (allowBoundary) {
            if ((hullX >= boundaryX) && ((hullX + hullW) <= (boundaryX + boundaryW))) {
                return (hullY >= boundaryY) && ((hullY + hullH) <= (boundaryY + boundaryH));
            }
        } else {
            if ((hullX > boundaryX) && ((hullX + hullW) < (boundaryX + boundaryW))) {
                return (hullY > boundaryY) && ((hullY + hullH) < (boundaryY + boundaryH));
            }
        }

        return false;
    }

    private boolean isAcceptable(Rect objectRect, Size frameSize) {

        double x = objectRect.x, y = objectRect.y;
        double w = objectRect.width, h = objectRect.height;
        double objectWhRatioMargin = (w / h) - ACCEPTABLE_OBJECT_WH_RATIO;
        double width = frameSize.width, height = frameSize.height;
        double objectAreaRatio = (w * h) / (width * height);

        logv("isAcceptable(): objectRect=" + objectRect + ", frameSize=" + frameSize + ", objectAreaRatio=" + objectAreaRatio);

        if ((-ACCEPTABLE_OBJECT_WH_RATIO_MARGIN <= objectWhRatioMargin) &&
                (objectWhRatioMargin <= ACCEPTABLE_OBJECT_WH_RATIO_MARGIN)) {
            float acceptable_ratio = ED_ONLY ? 0.7f : ACCEPTABLE_OBJECT_AREA_RATIO;
            if (objectAreaRatio <= acceptable_ratio) {

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

    private MatOfPoint _mergeHulls(@NonNull MatOfPoint newHull, MatOfPoint curMergedHull) {

        // Find ConvexHull including all points in CROP IMAGE
        List<Point> hullPoints = new ArrayList<>();
        if (curMergedHull != null) hullPoints.addAll(curMergedHull.toList());
        hullPoints.addAll(newHull.toList());
        return OpenCvUtils.convexHull(new MatOfPoint(hullPoints.toArray(new Point[0])), true);
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
            Log.v(LOG_TAG, "[" + mFrameIndexDf.format(frameIndex) + "] " + message);
        }
    }
}

