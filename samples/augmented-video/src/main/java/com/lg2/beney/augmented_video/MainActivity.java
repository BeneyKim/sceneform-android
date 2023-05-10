package com.lg2.beney.augmented_video;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.util.Pair;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentOnAttachListener;

import com.google.android.filament.Engine;
import com.google.android.filament.filamat.MaterialBuilder;
import com.google.android.filament.filamat.MaterialPackage;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.AugmentedImageDatabase;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.ImageFormat;
import com.google.ar.core.Plane;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Sceneform;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.EngineInstance;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInstance;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.BaseArFragment;
import com.google.ar.sceneform.ux.InstructionsController;
import com.google.ar.sceneform.ux.TransformableNode;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.lg2.beney.augmented_video.core.EdgeDetector;
import com.lg2.beney.augmented_video.core.IntermediateRecorder;
import com.lg2.beney.augmented_video.core.ObjectDetector;
import com.lg2.beney.augmented_video.core.VideoHash;
import com.lg2.beney.augmented_video.utils.ImageUtils;
import com.lg2.beney.augmented_video.utils.SnackbarHelper;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity implements
        FragmentOnAttachListener,
        BaseArFragment.OnSessionConfigurationListener {

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private static final boolean VDBG = false;

    private final List<CompletableFuture<Void>> futures = new ArrayList<>();
    private ArFragment arFragment;
    private boolean matrixDetected = false;
    private boolean rabbitDetected = false;
    private boolean math2Detected = false;
    private boolean math4Detected = false;
    @SuppressWarnings("FieldCanBeLocal")
    private AugmentedImageDatabase database;
    private Renderable plainVideoModel;
    private Material plainVideoMaterial;
    private MediaPlayer mediaPlayer;

    private final SnackbarHelper snackbarHelper = new SnackbarHelper();

    @SuppressWarnings("FieldCanBeLocal")
    private final String AR_IMAGE_DATABASE_FILENAME = "ar_image_database.imgdb";
    @SuppressWarnings("FieldCanBeLocal")
    private final String AR_IMAGE_DATABASE_JSON_FILENAME = "ar_image_database.json";

    private List<ArImageSet> mArImageSet = null;

    private ObjectDetector mObjectDetector = null;
    private EdgeDetector mEdgeDetector = null;
    private VideoHash mVideoHash = null;

    private ExecutorService mExecutorService;

    private static final boolean USE_BACKGROUND_4_OBJECT_DETECTION = true;

    private final DecimalFormat mFrameIndexDf = new DecimalFormat("00000000");

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(LOG_TAG, "static initializer: failed to init OpenCV");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        ViewCompat.setOnApplyWindowInsetsListener(toolbar, (v, insets) -> {
            ((ViewGroup.MarginLayoutParams) toolbar.getLayoutParams()).topMargin = insets
                    .getInsets(WindowInsetsCompat.Type.systemBars())
                    .top;

            return WindowInsetsCompat.CONSUMED;
        });
        getSupportFragmentManager().addFragmentOnAttachListener(this);

        if (savedInstanceState == null) {
            if (Sceneform.isSupported(this)) {
                getSupportFragmentManager().beginTransaction()
                        .add(R.id.arFragment, ArFragment.class, null)
                        .commit();
            }
        }

        if(Sceneform.isSupported(this)) {
            // .glb models can be loaded at runtime when needed or when app starts
            // This method loads ModelRenderable when app starts
            loadMatrixModel();
            loadMatrixMaterial();
        }

        mObjectDetector = new ObjectDetector(this);
        mEdgeDetector = new EdgeDetector(new IntermediateRecorder(this));
        mVideoHash = new VideoHash();

        if (USE_BACKGROUND_4_OBJECT_DETECTION) {
            mExecutorService = Executors.newFixedThreadPool(1);
        }
    }

    @Override
    public void onAttachFragment(@NonNull FragmentManager fragmentManager, @NonNull Fragment fragment) {
        if (fragment.getId() == R.id.arFragment) {
            arFragment = (ArFragment) fragment;
            arFragment.setOnSessionConfigurationListener(this);
        }
    }

    @Override
    public void onSessionConfiguration(Session session, Config config) {
        // Disable plane detection
        config.setPlaneFindingMode(Config.PlaneFindingMode.DISABLED);

        // Camera Resolution Up
        try {
            CameraConfigFilter filter = new CameraConfigFilter(session).setFacingDirection(CameraConfig.FacingDirection.BACK);
            List<CameraConfig> configs = new ArrayList<>(session.getSupportedCameraConfigs(filter));

            configs.sort((n1, n2) -> {
                if (n1.getImageSize().getWidth() != n2.getImageSize().getWidth()) {
                    return n2.getImageSize().getWidth() - n1.getImageSize().getWidth();
                }
                return n2.getImageSize().getHeight() - n1.getImageSize().getHeight();
            });

            if (VDBG) {
                Log.v(LOG_TAG, "Camera Config - " +
                        "getWidth: " + configs.get(0).getImageSize().getWidth() +
                        " ,getHeight: " + configs.get(0).getImageSize().getHeight() +
                        " ,getFpsRange: " + configs.get(0).getFpsRange() +
                        " ,getCameraId: " + configs.get(0).getCameraId());
            }

            session.setCameraConfig(configs.get(0));
            session.resume();
            session.pause();
            session.resume();

        } catch (CameraNotAvailableException e) {
            // Ignore
        }

        // Images to be detected by our AR need to be added in AugmentedImageDatabase
        // This is how database is created at runtime
        // You can also prebuild database in you computer and load it directly (see: https://developers.google.com/ar/develop/java/augmented-images/guide#database)

        try (InputStream is = getAssets().open(AR_IMAGE_DATABASE_FILENAME)) {
            database = AugmentedImageDatabase.deserialize(session, is);

        } catch (Exception e) {
            Log.e(LOG_TAG, "IO exception loading augmented image database.", e);

            database = new AugmentedImageDatabase(session);
            Bitmap matrixImage = BitmapFactory.decodeResource(getResources(), R.drawable.matrix);
            Bitmap rabbitImage = BitmapFactory.decodeResource(getResources(), R.drawable.rabbit);
            Bitmap math2Image = BitmapFactory.decodeResource(getResources(), R.drawable.math2);
            Bitmap math4Image = BitmapFactory.decodeResource(getResources(), R.drawable.math4);
            // Every image has to have its own unique String identifier
            database.addImage("matrix", matrixImage);
            database.addImage("rabbit", rabbitImage);
            database.addImage("math2", math2Image);
            database.addImage("math4", math4Image);
        }

        config.setAugmentedImageDatabase(database);

        // Check for image detection
        arFragment.setOnAugmentedImageUpdateListener(this::onAugmentedImageTrackingUpdate);
        arFragment.getArSceneView().getScene().addOnUpdateListener(this::onUpdateFrame);

        try (InputStream is = getAssets().open(AR_IMAGE_DATABASE_JSON_FILENAME)) {

            StringBuilder sb = new StringBuilder();
            BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
            String str;
            while ((str = br.readLine()) != null) {
                sb.append(str);
            }
            br.close();

            Gson gson = new Gson();

            Type listType = new TypeToken<ArrayList<ArImageSet>>(){}.getType();
            mArImageSet = gson.fromJson(sb.toString(), listType);

        } catch (Exception ignored) {
        }
    }

    private int frameIndex = -1;

    private void onUpdateFrame(FrameTime frameTime) {

        Frame frame = arFragment.getArSceneView().getArFrame();

        logv(frameIndex, "onUpdateFrame: frame time(ms) = " + frameTime.getDeltaTime(TimeUnit.MICROSECONDS));

        // If there is no frame or ARCore is not tracking yet, just return.
        if (frame == null || frame.getCamera().getTrackingState() != TrackingState.TRACKING) {
            return;
        }

        Collection<Plane> planes = frame.getUpdatedTrackables(Plane.class);
        for (Plane plane : planes) {
            // Find My Plane!
        }

        // Copy the camera stream to a bitmap

        // TODO: NEED TO OPTIMIZE, HOW TO DECIDE FRAME_DECODE_RATE?
        if (++frameIndex % 5 != 0) return;
        boolean decode = (frameIndex % 30 == 0);

        try (Image image = frame.acquireCameraImage()) {

            if (image.getFormat() != ImageFormat.YUV_420_888) {
                throw new IllegalArgumentException(
                        "Expected image in YUV_420_888 format, got format " + image.getFormat());
            }

            logv(frameIndex, "onUpdateFrame: image size = (" + image.getWidth() + "x" + image.getHeight() + ")");

            Mat bgrMat = ImageUtils.imageToBgrMat(image);

            if (USE_BACKGROUND_4_OBJECT_DETECTION) {
                mExecutorService.execute(new ObjectDetectTask(bgrMat, frameIndex, decode));

            } else {
                Rect objectBoundary = mObjectDetector.run(bgrMat.clone(), frameIndex, decode);
                log(frameIndex, "onUpdateFrame: objectBoundary=" + objectBoundary);

                if (decode) return;

                Pair<Mat, List<MatOfPoint>> result = mEdgeDetector.contours(bgrMat, frameIndex, true);
                // Mat cannyMat = result.first;
                List<MatOfPoint> contours = result.second;
                List<MatOfPoint> convexHulls = mEdgeDetector.convexHulls(bgrMat, contours, frameIndex, true);
                mEdgeDetector.mergeHulls(bgrMat, convexHulls, objectBoundary, frameIndex, true);

                Imgproc.drawContours(bgrMat,
                        contours,
                        -1, new Scalar(255, 0, 0), 10);

                ImageUtils.SaveBitmapToFile(this, ImageUtils.imageBgrMatToBitmap(bgrMat));
            }
        } catch (NotYetAvailableException notYetAvailableException) {
            // Ignore NotYetAvailableException
        } catch (Exception e) {
            loge("Exception copying image", e);
        }
    }

    class ObjectDetectTask implements Runnable {

        private final Mat mImage;
        private final int mFrameIndex;
        private final boolean mDecode;

        ObjectDetectTask(Mat image, int frameIndex, boolean decode) {

            mImage = image.clone();
            mFrameIndex = frameIndex;
            mDecode = decode;

            log(mFrameIndex, "ObjectDetectTask: mDecode=" + mDecode);

        }

        @Override
        public void run() {

            try {
                Rect objectBoundary = mObjectDetector.run(mImage.clone(), mFrameIndex, mDecode);
                log(mFrameIndex, "ObjectDetectTask: objectBoundary=" + objectBoundary);

                if (!mDecode) return;

                Pair<Mat, List<MatOfPoint>> result = mEdgeDetector.contours(mImage, mFrameIndex, true);
                // Mat cannyMat = result.first;
                List<MatOfPoint> contours = result.second;
                List<MatOfPoint> convexHulls = mEdgeDetector.convexHulls(mImage, contours, mFrameIndex, true);
                mEdgeDetector.mergeHulls(mImage, convexHulls, objectBoundary, mFrameIndex, true);

                Imgproc.drawContours(mImage,
                        contours,
                        -1, new Scalar(255, 0, 0), 10);

                // ImageUtils.SaveBitmapToFile(this, ImageUtils.imageBgrMatToBitmap(bgrMat));

            } catch (Exception e) {
                loge("Exception detect object", e);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        futures.forEach(future -> {
            if (!future.isDone())
                future.cancel(true);
        });

        if (mediaPlayer != null) {
            mediaPlayer.stop();
            mediaPlayer.reset();
        }
    }

    private void loadMatrixModel() {
        futures.add(ModelRenderable.builder()
                .setSource(this, Uri.parse("models/Video.glb"))
                .setIsFilamentGltf(true)
                .build()
                .thenAccept(model -> {
                    //removing shadows for this Renderable
                    model.setShadowCaster(false);
                    model.setShadowReceiver(true);
                    plainVideoModel = model;
                })
                .exceptionally(
                        throwable -> {
                            Toast.makeText(this, "Unable to load renderable", Toast.LENGTH_LONG).show();
                            return null;
                        }));
    }

    private void loadMatrixMaterial() {
        Engine filamentEngine = EngineInstance.getEngine().getFilamentEngine();

        MaterialBuilder.init();
        MaterialBuilder materialBuilder = new MaterialBuilder()
                .platform(MaterialBuilder.Platform.MOBILE)
                .name("External Video Material")
                .require(MaterialBuilder.VertexAttribute.UV0)
                .shading(MaterialBuilder.Shading.UNLIT)
                .doubleSided(true)
                .samplerParameter(MaterialBuilder.SamplerType.SAMPLER_EXTERNAL, MaterialBuilder.SamplerFormat.FLOAT, MaterialBuilder.ParameterPrecision.DEFAULT, "videoTexture")
                .optimization(MaterialBuilder.Optimization.NONE);

        MaterialPackage plainVideoMaterialPackage = materialBuilder
                .blending(MaterialBuilder.BlendingMode.OPAQUE)
                .material("void material(inout MaterialInputs material) {\n" +
                        "    prepareMaterial(material);\n" +
                        "    material.baseColor = texture(materialParams_videoTexture, getUV0()).rgba;\n" +
                        "}\n")
                .build(filamentEngine);
        if (plainVideoMaterialPackage.isValid()) {
            ByteBuffer buffer = plainVideoMaterialPackage.getBuffer();
            futures.add(Material.builder()
                    .setSource(buffer)
                    .build()
                    .thenAccept(material -> plainVideoMaterial = material)
                    .exceptionally(
                            throwable -> {
                                Toast.makeText(this, "Unable to load material", Toast.LENGTH_LONG).show();
                                return null;
                            }));
        }
        MaterialBuilder.shutdown();
    }

    public void onAugmentedImageTrackingUpdate(AugmentedImage augmentedImage) {
        // If there are both images already detected, for better CPU usage we do not need scan for them

        Log.d(LOG_TAG, "onAugmentedImageTrackingUpdate(): detected name = " + augmentedImage.getName());

        if (matrixDetected && rabbitDetected && math2Detected && math4Detected) {
            return;
        }

        if (augmentedImage.getTrackingState() == TrackingState.TRACKING
                && augmentedImage.getTrackingMethod() == AugmentedImage.TrackingMethod.FULL_TRACKING) {

            // Setting anchor to the center of Augmented Image
            AnchorNode anchorNode = new AnchorNode(augmentedImage.createAnchor(augmentedImage.getCenterPose()));

            // If matrix video haven't been placed yet and detected image has String identifier of "matrix"
            if (!matrixDetected && augmentedImage.getName().equals("matrix")) {
                matrixDetected = true;
                Toast.makeText(this, "Matrix tag detected", Toast.LENGTH_LONG).show();

                // AnchorNode placed to the detected tag and set it to the real size of the tag
                // This will cause deformation if your AR tag has different aspect ratio than your video
                anchorNode.setWorldScale(new Vector3(augmentedImage.getExtentX(), 1f, augmentedImage.getExtentZ()));
                arFragment.getArSceneView().getScene().addChild(anchorNode);

                TransformableNode videoNode = new TransformableNode(arFragment.getTransformationSystem());
                // For some reason it is shown upside down so this will rotate it correctly
                videoNode.setLocalRotation(Quaternion.axisAngle(new Vector3(0, 1f, 0), 180f));
                anchorNode.addChild(videoNode);

                // Setting texture
                ExternalTexture externalTexture = new ExternalTexture();
                RenderableInstance renderableInstance = videoNode.setRenderable(plainVideoModel);
                renderableInstance.setMaterial(plainVideoMaterial);

                // Setting MediaPLayer
                renderableInstance.getMaterial().setExternalTexture("videoTexture", externalTexture);
                mediaPlayer = MediaPlayer.create(this, R.raw.matrix);
                mediaPlayer.setLooping(true);
                mediaPlayer.setSurface(externalTexture.getSurface());
                mediaPlayer.start();
            }
            // If rabbit model haven't been placed yet and detected image has String identifier of "rabbit"
            // This is also example of model loading and placing at runtime
            if (!rabbitDetected && augmentedImage.getName().equals("rabbit")) {
                rabbitDetected = true;
                Toast.makeText(this, "Rabbit tag detected", Toast.LENGTH_LONG).show();

                // anchorNode.setWorldScale(new Vector3(3.5f, 3.5f, 3.5f));
                // anchorNode.setWorldScale(new Vector3(0.4f, 0.4f, 0.4f));
                anchorNode.setWorldScale(new Vector3(0.03f, 0.03f, 0.03f)); // tyrannosaurus_rex
                // anchorNode.setWorldScale(new Vector3(0.0003f, 0.0003f, 0.0003f)); // f18
                arFragment.getArSceneView().getScene().addChild(anchorNode);

                futures.add(ModelRenderable.builder()
                        .setSource(this, Uri.parse("models/tyrannosaurus_rex.glb"))
                        .setIsFilamentGltf(true)
                        .build()
                        .thenAccept(rabbitModel -> {
                            TransformableNode modelNode = new TransformableNode(arFragment.getTransformationSystem());
                            modelNode.setRenderable(rabbitModel).animate(true).start();
                            anchorNode.addChild(modelNode);
                        })
                        .exceptionally(
                                throwable -> {
                                    Toast.makeText(this, "Unable to load rabbit model", Toast.LENGTH_LONG).show();
                                    return null;
                                }));
            }

            if (!math2Detected && augmentedImage.getName().equals("math2")) {
                math2Detected = true;
                Toast.makeText(this, "Math2 tag detected", Toast.LENGTH_LONG).show();

                // anchorNode.setWorldScale(new Vector3(3.5f, 3.5f, 3.5f));
                // anchorNode.setWorldScale(new Vector3(0.4f, 0.4f, 0.4f));
                // anchorNode.setWorldScale(new Vector3(0.03f, 0.03f, 0.03f)); // tyrannosaurus_rex
                anchorNode.setWorldScale(new Vector3(0.0001f, 0.0001f, 0.0001f)); // f18
                arFragment.getArSceneView().getScene().addChild(anchorNode);

                futures.add(ModelRenderable.builder()
                        .setSource(this, Uri.parse("models/f18.glb"))
                        .setIsFilamentGltf(true)
                        .build()
                        .thenAccept(rabbitModel -> {
                            TransformableNode modelNode = new TransformableNode(arFragment.getTransformationSystem());
                            modelNode.setRenderable(rabbitModel).animate(true).start();
                            anchorNode.addChild(modelNode);
                        })
                        .exceptionally(
                                throwable -> {
                                    Toast.makeText(this, "Unable to load f18 model", Toast.LENGTH_LONG).show();
                                    return null;
                                }));
            }

            if (!math4Detected && augmentedImage.getName().equals("math4")) {
                math4Detected = true;
                Toast.makeText(this, "Math4 tag detected", Toast.LENGTH_LONG).show();

                // anchorNode.setWorldScale(new Vector3(3.5f, 3.5f, 3.5f));
                anchorNode.setWorldScale(new Vector3(0.2f, 0.2f, 0.2f));
                // anchorNode.setWorldScale(new Vector3(0.03f, 0.03f, 0.03f)); // tyrannosaurus_rex
                // anchorNode.setWorldScale(new Vector3(0.0003f, 0.0003f, 0.0003f)); // f18
                arFragment.getArSceneView().getScene().addChild(anchorNode);

                futures.add(ModelRenderable.builder()
                        .setSource(this, Uri.parse("models/shiba.glb"))
                        .setIsFilamentGltf(true)
                        .build()
                        .thenAccept(rabbitModel -> {
                            TransformableNode modelNode = new TransformableNode(arFragment.getTransformationSystem());
                            modelNode.setRenderable(rabbitModel).animate(true).start();
                            anchorNode.addChild(modelNode);
                        })
                        .exceptionally(
                                throwable -> {
                                    Toast.makeText(this, "Unable to load shiba model", Toast.LENGTH_LONG).show();
                                    return null;
                                }));
            }
        }

        if (matrixDetected && rabbitDetected && math2Detected && math4Detected) {
            arFragment.getInstructionsController().setEnabled(
                    InstructionsController.TYPE_AUGMENTED_IMAGE_SCAN, false);
        }
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

    @SuppressWarnings("unused")
    private void loge(String message) {
        Log.e(LOG_TAG, message);
    }

    @SuppressWarnings("unused")
    private void loge(String message, Throwable throwable) {
        Log.e(LOG_TAG, message, throwable);
    }
}