package com.lg2.beney.augmented_paper_detector.core;

import android.os.Environment;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import com.lg2.beney.augmented_paper_detector.utils.TypeConvertor;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.img_hash.ImgHashBase;
import org.opencv.img_hash.PHash;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("unused")
public class VideoHash {

    private static final String LOG_TAG = VideoHash.class.getSimpleName();

    private static final int HASH_HPF = 10; // Hash Per Frame
    private static final float DROP_BOUNDARY_RATIO = 0.05f; // 이미지 외곽의 부분을 버리고 해쉬를 생성
    private static final Size HASH_IMAGE_SIZE = new Size(100, 100); // 이미지 해쉬를 생성할 때 사용할 이미지 크기
    private static final int NUMBER_OF_COMPARE_OUTPUT = 5;
    private static final int HASH_DIFFERENCE_THRESHOLD = 14;

    private final String mExternalStoragePath;
    private final ImgHashBase mHashAlgorithm;
    private final Map<Integer, Mat> mHashDataSet;

    private final DecimalFormat mFrameIndexDf = new DecimalFormat("00000000");
    private final Gson mGson;

    public VideoHash() {
        this(null);
    }

    @SuppressWarnings("ConstantConditions")
    public VideoHash(ImgHashBase hashAlgorithm) {

        mExternalStoragePath = Environment.getExternalStorageDirectory().getPath();

        if (hashAlgorithm == null) {
            mHashAlgorithm = hashAlgorithm;
        } else {
            mHashAlgorithm = PHash.create();
        }

        mHashDataSet = new HashMap<>();
        mGson = new GsonBuilder().setPrettyPrinting().create();
    }

    public boolean compute(String fileName) {
        return this.compute(fileName, 0, -1, "data.json");
    }

    public boolean compute(String fileName, int startFrameIndex) {
        return this.compute(fileName, startFrameIndex, -1, "data.json");
    }

    public boolean compute(String fileName, int startFrameIndex, int endFrameIndex) {
        return this.compute(fileName, startFrameIndex, endFrameIndex, "data.json");
    }

    public boolean compute(String fileName, int startFrameIndex, int endFrameIndex, String outputFileName) {

        if (fileName == null || fileName.isEmpty()) {
            log("compute(): argument fileName(" + fileName + ") is invalid");
            return false;
        }

        File inputFile = new File(mExternalStoragePath + fileName);

        if (!inputFile.exists()) {
            log("compute():" + fileName + " does not exist.");
            return false;
        }

        if (!inputFile.isFile()) {
            log("compute():" + fileName + " is not file.");
            return false;
        }

        if (startFrameIndex < 0) {
            log("compute(): startFrameIndex(" + startFrameIndex + ") is invalid.");
            return false;
        }

        if ((endFrameIndex != -1) && (startFrameIndex > endFrameIndex)) {
            log("compute(): endFrameIndex(" + endFrameIndex + ") must be bigger than startFrameIndex( " + startFrameIndex + ").");
            return false;
        }

        // Open Video File
        VideoCapture videoInput = new VideoCapture(mExternalStoragePath + fileName);

        if (!videoInput.isOpened()) {
            log("compute(): failed to open " + fileName + ".");
            return false;
        }

        int frameCount = (int) videoInput.get(Videoio.CAP_PROP_FRAME_COUNT);

        if (startFrameIndex >= frameCount) {
            log("compute(): startFrameIndex(" + startFrameIndex + ") must be less than frame_count(" + frameCount + ").");
            return false;
        }

        if (endFrameIndex == -1) {
            endFrameIndex = frameCount;
        }

        if (endFrameIndex > frameCount) {
            endFrameIndex = frameCount;
        }

        if (outputFileName == null || outputFileName.isEmpty()) {
            log("compute(): argument outputFileName(" + outputFileName + ") is invalid");
            return false;
        }

        return compute(videoInput, startFrameIndex, endFrameIndex, outputFileName);
    }

    private boolean compute(VideoCapture videoInput, int startFrameIndex, int endFrameIndex, String outputFileName) {

        int index = 0;
        int frameIndex = startFrameIndex;
        videoInput.set(Videoio.CAP_PROP_POS_FRAMES, startFrameIndex);

        Map<Integer, Long> outputValues = new HashMap<>();
        mHashDataSet.clear();

        while (videoInput.isOpened() && frameIndex < endFrameIndex) {

            Mat frame = new Mat();
            boolean frameIsRead = videoInput.read(frame);

            if (frameIsRead) {

                if (index % HASH_HPF != 0) {
                    index += 1;
                    frameIndex += 1;
                    continue;
                }

                Imgcodecs.imwrite(mExternalStoragePath + "/" + frameIndex + ".jpg", frame);

                int height = frame.height(), width = frame.width();

                int heightS = (int) (height * DROP_BOUNDARY_RATIO);
                int heightE = (int) (height * (1.0 - DROP_BOUNDARY_RATIO));
                int widthS = (int) (width * DROP_BOUNDARY_RATIO);
                int widthE = (int) (width * (1.0 - DROP_BOUNDARY_RATIO));
                frame = new Mat(frame, new Rect(widthS, heightS, widthE, heightE));
                Imgproc.resize(frame, frame, HASH_IMAGE_SIZE);


                Mat hashArray = new Mat();
                mHashAlgorithm.compute(frame, hashArray);
                mHashDataSet.put(frameIndex, hashArray);

                byte[] bytesHashArray = new byte[(int) hashArray.total() * hashArray.channels()];
                hashArray.get(0, 0, bytesHashArray);

                long hashValue = TypeConvertor.bytesToLong(bytesHashArray);
                outputValues.put(frameIndex, hashValue);

                log("compute(): frameIndex(" + mFrameIndexDf.format(frameIndex) +
                        ") === hashArray ===> " + hashArray +
                        " === hashValue ===> " + hashValue);
            } else {
                break;
            }

            index += 1;
            frameIndex += 1;
        }

        try {
            mGson.toJson(outputValues, new FileWriter(mExternalStoragePath + "/" + outputFileName));
        } catch (IOException e) {
            loge("Failed to save HashDataSet to file", e);
        }

        return true;
    }

    private boolean load(String inputPath) {
        return this.load(inputPath, "data.json");
    }

    @SuppressWarnings("SameParameterValue")
    private boolean load(String inputPath, String inputFileName) {

        if (inputPath == null || inputPath.isEmpty()) {
            log("load(): inputPath(" + inputPath + ") is invalid");
            return false;
        }

        if (inputFileName == null || inputFileName.isEmpty()) {
            log("load(): inputFileName(" + inputFileName + ") is invalid");
            return false;
        }

        String inputFullPath = mExternalStoragePath + "/" + inputPath + "/" + inputFileName;
        File inputDir = new File(mExternalStoragePath + "/" + inputPath);
        File inputFile = new File(inputFullPath);

        if (!inputFile.exists()){
            if (!inputDir.exists()) {
                log("load(): inputPath(" + inputPath + ") does not exist");
            } else{
                log("load(): inputFileName(" + inputFileName + ") does not exist");
            }
            return false;
        }

        if (!inputFile.isFile()){
            log("load(): inputFileName(" + inputFileName + ") exists and it's not file.");
            return false;

        }

        return load(inputFile);
    }

    private boolean load(File inputFile) {

        final Map<Integer, Long> inputHashDataset;
        mHashDataSet.clear();

        try {
            JsonReader reader = new JsonReader(new FileReader(inputFile));
            inputHashDataset = mGson.fromJson(reader, new TypeToken<Map<Integer, Long>>(){}.getType());
            if (inputHashDataset == null) {
                return false;
            }

        } catch (FileNotFoundException e) {
            loge("load(): inputFileName (" + inputFile.getName() + ") does not exists and it's not file.", e);
            return false;
        }

        inputHashDataset.keySet().forEach(frameIndex -> {
            Long hashValue = inputHashDataset.get(frameIndex);
            if (hashValue != null) {
                ByteBuffer byteBuffer = ByteBuffer.allocate(Long.BYTES);
                byteBuffer.putLong(hashValue);
                Mat hashArray = new Mat(1, 8, CvType.CV_8U, byteBuffer);
                hashArray.reshape(1, 1);

                log("load(): frameIndex(" + mFrameIndexDf.format(frameIndex) +
                        ") === hashValue ===> " + hashValue +
                        " === hashArray ===> " + hashArray);
                mHashDataSet.put(frameIndex, hashArray);
            }
        });

        return true;
    }

    private List<VideoHashResult> compare(String inputPath, String inputFileName) {
        return compare(inputPath, inputFileName, -1);
    }

    @SuppressWarnings("SameParameterValue")
    private List<VideoHashResult> compare(String inputPath, String inputFileName, int differenceThreshold) {

        if (inputPath == null || inputPath.isEmpty()) {
            log("compare() inputPath(" + inputPath + ") is invalid");
            return null;
        }

        if (inputFileName == null || inputFileName.isEmpty()) {
            log("compare(): input_filename(" + inputFileName + ") is invalid");
            return null;
        }

        String inputFullPath = mExternalStoragePath + "/" + inputPath + "/" + inputFileName;
        File inputDir = new File(mExternalStoragePath + "/" + inputPath);
        File inputFile = new File(inputFullPath);

        if (!inputFile.exists()){
            if (!inputDir.exists()) {
                log("compare(): inputPath(" + inputPath + ") does not exist");
            } else{
                log("compare(): inputFileName(" + inputFileName + ") does not exist");
            }
            return null;
        }


        if (!inputFile.isFile()){
            log("compare(): inputFileName(" + inputFileName + ") exists and it's not file.");
            return null;
        }

        if (differenceThreshold == -1) {
            differenceThreshold = HASH_DIFFERENCE_THRESHOLD;
        }

        Mat inputImage = Imgcodecs.imread(inputFullPath);

        return compare(inputImage, differenceThreshold);
    }

    private List<VideoHashResult> compare2(Mat inputImage) {
        return compare2(inputImage, -1);
    }

    @SuppressWarnings("SameParameterValue")
    private List<VideoHashResult> compare2(Mat inputImage, int differenceThreshold) {

        if (differenceThreshold == -1) {
            differenceThreshold = HASH_DIFFERENCE_THRESHOLD;
        }

        return compare(inputImage, differenceThreshold);
    }

    private List<VideoHashResult> compare(Mat inputImage, int differenceThreshold) {

        if (mHashDataSet == null) {
            log("compare(): hash_dataset is invalid");
            return null;
        }

        if (mHashDataSet.size() == 0) {
            log("compare(): hash_dataset is empty");
            return null;
        }

        int height = inputImage.height(), width = inputImage.width();

        int heightS = (int) (height * DROP_BOUNDARY_RATIO);
        int heightE = (int) (height * (1.0 - DROP_BOUNDARY_RATIO));
        int widthS = (int) (width * DROP_BOUNDARY_RATIO);
        int widthE = (int) (width * (1.0 - DROP_BOUNDARY_RATIO));

        inputImage = new Mat(inputImage, new Rect(widthS, heightS, widthE, heightE));
        Imgproc.resize(inputImage, inputImage, HASH_IMAGE_SIZE);

        Mat hashArray = new Mat();
        mHashAlgorithm.compute(inputImage, hashArray);

        final List<VideoHashResult> differenceValues = new ArrayList<>();
        mHashDataSet.forEach((frameIndex, frameHashArray) -> {
            int difference = (int) mHashAlgorithm.compare(hashArray, frameHashArray);
            if (difference <= HASH_DIFFERENCE_THRESHOLD){
                differenceValues.add(new VideoHashResult(frameIndex, difference));
            }
        });

        differenceValues.sort(Comparator.comparingInt(t -> t.difference));

        return differenceValues.size() > NUMBER_OF_COMPARE_OUTPUT ?
                differenceValues.subList(0, NUMBER_OF_COMPARE_OUTPUT) : differenceValues;
    }

    private void log(String message) {
        Log.d(LOG_TAG, message);
    }

    private void logv(String message) {
        Log.v(LOG_TAG, message);
    }

    private void loge(String message) {
        Log.e(LOG_TAG, message);
    }

    private void loge(String message, Throwable throwable) {
        Log.e(LOG_TAG, message, throwable);
    }

    public static class VideoHashResult implements Comparator<VideoHashResult> {

        int frameIndex;
        int difference;

        VideoHashResult(int frameIndex, int difference) {
            this.frameIndex = frameIndex;
            this.difference = difference;
        }

        @Override
        public int hashCode() {
            return Integer.hashCode(frameIndex) << 4 + Integer.hashCode(difference);
        }

        @Override
        public int compare(VideoHashResult t1, VideoHashResult t2) {
            return Integer.compare(t1.difference, t2.difference);
        }
    }
}
