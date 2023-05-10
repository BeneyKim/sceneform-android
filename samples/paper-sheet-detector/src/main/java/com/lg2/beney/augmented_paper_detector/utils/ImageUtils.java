package com.lg2.beney.augmented_paper_detector.utils;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.Image;
import android.net.Uri;
import android.os.Environment;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import androidx.annotation.NonNull;

import com.lg2.beney.augmented_video.utils.YuvToRgbConverter;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class ImageUtils {

    private static final String LOG_TAG = ImageUtils.class.getSimpleName();

    private static int pixelCount = -1;

    public static Bitmap convertYuv(Activity activity, Image image) {
        YuvToRgbConverter yuvConverter = new YuvToRgbConverter(activity.getApplicationContext());
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        return yuvConverter.yuvToRgb(image, bitmap);
    }

    public static Bitmap getBitmapFromImage(Activity activity, Image image,  int imageRotation) {
        Bitmap bitmap = convertYuvtoRgb(activity, image);
        return rotateBitmap(bitmap, imageRotation);
    }

    public static byte[] getByteArrayFromBitmap(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        // Convert the bitmap to a JPEG
        // Just in case it's a format that Android understands but Cloud Vision
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int rotation) {
        if (rotation == 0) return bitmap;

        Matrix matrix = new Matrix();
        matrix.postRotate((float)rotation);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, false);
    }

    public static Bitmap convertYuvtoRgb(Activity activity, Image image) {
        // Change bitmap from image
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);

        // Get the YUV data in byte array form using NV21 format
        pixelCount = image.getWidth() * image.getWidth();
        int pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888);
        byte[] yuvBuffer = new byte[pixelCount * pixelSizeBits / 8];

        imageToByteArray(image, yuvBuffer);

        RenderScript rs = RenderScript.create(activity.getApplicationContext());
        ScriptIntrinsicYuvToRGB scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

        Type yuvType = new Type.Builder(rs, Element.YUV(rs)).setYuvFormat(ImageFormat.NV21).create();
        Allocation in = Allocation.createSized(rs, yuvType.getElement(), yuvBuffer.length);
        Allocation out = Allocation.createFromBitmap(rs, bitmap);

        // Convert NV21 format YUV to RGB
        in.copyFrom(yuvBuffer);
        scriptYuvToRgb.setInput(in);
        scriptYuvToRgb.forEach(out);
        out.copyTo(bitmap);

        return bitmap;
    }

    public static void imageToByteArray(Image image, byte[] outputBuffer) {
        if (image.getFormat() != ImageFormat.YUV_420_888) return;

        Rect imageCrop = new Rect(0, 0, image.getWidth(), image.getHeight());
        Image.Plane[] imagePlanes = image.getPlanes();

        int outputStride = 0;
        int outputOffset = 0;
        for (int planeIndex = 0; planeIndex < imagePlanes.length; planeIndex++) {
            switch (planeIndex) {
                case 0:
                    outputStride = 1;
                    outputOffset = 0;
                    break;
                case 1:
                    outputStride = 2;
                    // For NV21 format, U is in odd-numbered indices
                    outputOffset = pixelCount + 1;
                    break;
                case 2:
                    outputStride = 2;
                    // For NV21 format, V is in even-numbered indices
                    outputOffset = pixelCount;
                    break;
                default:
                    // Image contains more than 3 planes, something strange is going on
                    return;
            }

            ByteBuffer planeBuffer = imagePlanes[planeIndex].getBuffer();
            int rowStride = imagePlanes[planeIndex].getRowStride();
            int pixelStride = imagePlanes[planeIndex].getPixelStride();

            // We have to divide the width and height by two if it's not the Y plane
            Rect planeCrop;
            if (planeIndex == 0) {
                planeCrop = imageCrop;
            } else {
                planeCrop = new Rect(imageCrop.left / 2, imageCrop.top / 2, imageCrop.right / 2, imageCrop.bottom / 2);
            }

            int planeWidth = planeCrop.width();
            int planeHeight = planeCrop.height();

            // Intermediate buffer used to store the bytes of each row
            byte[] rowBuffer = new byte[imagePlanes[planeIndex].getRowStride()];

            // Size of each row in bytes
            int rowLength = 0;
            if (pixelStride == 1 && outputStride == 1) {
                rowLength = planeWidth;
            } else {
                // Take into account that the stride may include data from pixels other than this
                // particular plane and row, and that could be between pixels and not after every
                // pixel:
                //
                // |---- Pixel stride ----|                    Row ends here --> |
                // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
                //
                // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
                rowLength = (planeWidth - 1) * pixelStride + 1;
            }

            for (int row = 0; row < planeHeight; row++) {
                // Move buffer position to the beginning of this row
                planeBuffer.position((row + planeCrop.top) * rowStride + planeCrop.left * pixelStride);

                if (pixelStride == 1 && outputStride == 1) {
                    // When there is a single stride value for pixel and output, we can just copy
                    // the entire row in a single step
                    planeBuffer.get(outputBuffer, outputOffset, rowLength);
                    outputOffset += rowLength;
                } else {
                    // When either pixel or output have a stride > 1 we must copy pixel by pixel
                    planeBuffer.get(rowBuffer, 0, rowLength);
                    for (int col = 0; col < planeWidth; col++) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride];
                        outputOffset += outputStride;
                    }
                }
            } // for
        } // for
    }

    /**
     * Takes an Android Image in the YUV_420_888 format and returns an OpenCV Mat.
     *
     * @param image Image in the YUV_420_888 format.
     * @return OpenCV Yuv Mat.
     */
    public static Mat imageToMat(Image image) {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    // Advance buffer the remainder of the row stride, unless on the last row.
                    // Otherwise, this will throw an IllegalArgumentException because the buffer
                    // doesn't include the last padding.
                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {

                    // On the last row only read the width of the image minus the pixel stride
                    // plus one. Otherwise, this will throw a BufferUnderflowException because the
                    // buffer doesn't include the last padding.
                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }

                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        // Finally, create the Mat.
        Mat mat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        mat.put(0, 0, data);

        return mat;
    }

    public static Mat imageToBgrMat(Image image) {
        Mat mYuvMat = imageToMat(image);

        Mat bgrMat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC4);
        Imgproc.cvtColor(mYuvMat, bgrMat, Imgproc.COLOR_YUV2BGR_I420);

        return bgrMat;
    }

    public static Mat imageToRgbMat(Image image) {
        Mat mYuvMat = imageToMat(image);

        Mat rgbMat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        Imgproc.cvtColor(mYuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_I420);

        return rgbMat;
    }

    public static Bitmap imageBgrMatToBitmap(Mat bgrMat) {

        Mat rgbaMatOut = new Mat();
        Imgproc.cvtColor(bgrMat, rgbaMatOut, Imgproc.COLOR_BGR2RGBA, 0);
        final Bitmap bitmap = Bitmap.createBitmap(bgrMat.cols(), bgrMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgbaMatOut, bitmap);

        return bitmap;
    }

    public static Bitmap imageRgbMatToBitmap(Mat rgbMat) {

        final Bitmap bitmap = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgbMat, bitmap);

        return bitmap;
    }

    private static String currentPhotoPath;

    private static File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.KOREA).format(new Date());
        String imageFileName = timeStamp + "_";
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

    public static String SaveBitmapToFile(@NonNull Context context, Bitmap bitmap) {

        OutputStream out = null;

        try {
            File fileCacheItem = createImageFile();
            out = new FileOutputStream(fileCacheItem);

            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);

            galleryAddPic(context);

            return currentPhotoPath;

        } catch (Exception e) {
            Log.e(LOG_TAG, "SaveBitmapToFile, error = " + currentPhotoPath, e);
            e.printStackTrace();
        } finally {
            try {
                if (out != null) {
                    out.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return null;
    }

    private static void galleryAddPic(@NonNull Context context) {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(currentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        context.sendBroadcast(mediaScanIntent);
    }

}
