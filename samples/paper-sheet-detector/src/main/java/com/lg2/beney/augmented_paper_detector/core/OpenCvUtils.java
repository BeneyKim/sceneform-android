package com.lg2.beney.augmented_paper_detector.core;

import androidx.core.util.Pair;

import org.opencv.core.CvType;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.stream.IntStream;

public class OpenCvUtils {

    private static final String LOG_TAG = OpenCvUtils.class.getSimpleName();

    private static MatOfPoint2f _approxPolyDP(MatOfPoint matOfPoint, double epsilon, boolean closed) {

        MatOfPoint2f inputPoints = new MatOfPoint2f(matOfPoint.toArray());
        MatOfPoint2f outputPoints = new MatOfPoint2f();

        double perimeter = Imgproc.arcLength(inputPoints, closed);
        Imgproc.approxPolyDP(inputPoints, outputPoints, epsilon * perimeter, closed);

        return outputPoints;
    }

    public static MatOfPoint approxPolyDP(MatOfPoint matOfPoint, double epsilon, boolean closed) {

        MatOfPoint2f outputPoints = _approxPolyDP(matOfPoint, epsilon, closed);

        MatOfPoint approxPoint = new MatOfPoint();
        outputPoints.convertTo(approxPoint, CvType.CV_32S);

        return approxPoint;
    }

    public static Pair<MatOfPoint, Double> approxPolyDPwArea(MatOfPoint matOfPoint, double epsilon, boolean closed) {

        MatOfPoint2f outputPoints = _approxPolyDP(matOfPoint, epsilon, closed);

        MatOfPoint approxPoint = new MatOfPoint();
        outputPoints.convertTo(approxPoint, CvType.CV_32S);

        RotatedRect minAreaRect = Imgproc.minAreaRect(outputPoints);
        double minArea = minAreaRect.size.area();

        return Pair.create(approxPoint, minArea);
    }

    public static MatOfPoint convexHull(MatOfPoint matOfPoint) {
        return convexHull(matOfPoint, true);
    }

    public static MatOfPoint convexHull(MatOfPoint matOfPoint, boolean clockwise) {
        List<Point> allPoints = matOfPoint.toList();
        MatOfInt convexHullIdxs = new MatOfInt();
        Imgproc.convexHull(matOfPoint, convexHullIdxs, clockwise);
        return new MatOfPoint(IntStream.of(convexHullIdxs.toArray()).mapToObj(allPoints::get).toArray(Point[]::new));
    }
}
