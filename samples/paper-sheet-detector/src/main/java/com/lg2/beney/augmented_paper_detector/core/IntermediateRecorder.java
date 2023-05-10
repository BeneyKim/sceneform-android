package com.lg2.beney.augmented_paper_detector.core;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

@SuppressWarnings("unused")
public class IntermediateRecorder {

    private final Context mContext;

    public IntermediateRecorder(Context context) {
        mContext = context;
    }

    private static final String LOG_TAG = IntermediateRecorder.class.getSimpleName();

    public void saveImage(Mat frame, int frame_index){
        this.saveImage(frame, frame_index, 0, -1, null, "intermediate");
    }

    public void saveImage(Mat frame, int frame_index, int step_index){
        this.saveImage(frame, frame_index, step_index, -1, null, "intermediate");
    }

    public void saveImage(Mat frame, int frame_index, int step_index, int sub_step_index){
        this.saveImage(frame, frame_index, step_index, sub_step_index, null, "intermediate");
    }

    public void saveImage(Mat frame, int frame_index, int step_index, String keyword){
        this.saveImage(frame, frame_index, step_index, -1, null, keyword);
    }

    public void saveImage(Mat frame, int frame_index, int step_index, int sub_step_index, Rect crop_rect){
        this.saveImage(frame, frame_index, step_index, sub_step_index, crop_rect, "intermediate");
    }

    public void saveImage(Mat frame, int frame_index, int step_index, int sub_step_index, String keyword){
        this.saveImage(frame, frame_index, step_index, sub_step_index, null, keyword);
    }

    public void saveImage(Mat frame, int frame_index, int step_index, int sub_step_index, Rect crop_rect, String keyword){

        String fileName;
        if (sub_step_index == -1) {
            fileName = "" + frame_index + "-" + step_index + "-" + keyword;
        } else {
            fileName = "" + frame_index + "-" + step_index + "-" + sub_step_index + "-" + keyword;
        }

        Bitmap bitmap = imageBgrMatToBitmap(frame);
        SaveBitmapToFile(fileName, bitmap);
    }

    public Bitmap imageBgrMatToBitmap(Mat bgrMat) {

        Mat rgbaMatOut = new Mat();
        Imgproc.cvtColor(bgrMat, rgbaMatOut, Imgproc.COLOR_BGR2RGBA, 0);
        final Bitmap bitmap = Bitmap.createBitmap(bgrMat.cols(), bgrMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgbaMatOut, bitmap);

        return bitmap;
    }

    private static String currentPhotoPath;
    private static File createImageFile(String fileName) throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.KOREA).format(new Date());
        String imageFileName = timeStamp + "_" + fileName + "_";
        File storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    public void SaveBitmapToFile(String fileName, Bitmap bitmap) {

        OutputStream out = null;

        try {
            File fileCacheItem = createImageFile(fileName);
            out = new FileOutputStream(fileCacheItem);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            galleryAddPic();

        } catch (Exception e) {
            Log.e(LOG_TAG, "SaveBitmapToFile, error = " + currentPhotoPath, e);
        } finally {
            try {
                if (out != null) {
                    out.close();
                }
            } catch (IOException e) {
                Log.e(LOG_TAG, "SaveBitmapToFile, error = " + currentPhotoPath, e);
            }
        }
    }

    private void galleryAddPic() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(currentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        mContext.sendBroadcast(mediaScanIntent);
    }
}
