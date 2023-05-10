package com.lg2.beney.augmented_paper_detector;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.img_hash.AverageHash;
import org.opencv.img_hash.ImgHashBase;

import java.nio.ByteBuffer;

public class ArImageInfo {

    private static final ImgHashBase Hasher = AverageHash.create();

    public int TimestampInMs;

    public long Hash;

    public int Score;

    public boolean IsSimilar(long hash)
    {
        ByteBuffer lBuffer = ByteBuffer.allocate(Long.BYTES);
        ByteBuffer rBuffer = ByteBuffer.allocate(Long.BYTES);

        lBuffer.putLong(this.Hash);
        rBuffer.putLong(hash);

        Mat lMat = new Mat(1, 8, CvType.CV_8U, lBuffer);
        Mat rMat = new Mat(1, 8, CvType.CV_8U, rBuffer);

        lMat = lMat.reshape(1, 1);
        rMat = rMat.reshape(1, 1);

        return Hasher.compare(lMat, rMat) < Constants.IMAGE_HASH_THRESHOLD;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("{")
                .append("Hash=").append(Hash)
                .append(", TimeStamp=").append(TimestampInMs)
                .append(", Score=").append(Score).append("}");
        return sb.toString();
    }
}
