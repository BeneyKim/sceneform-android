package com.lg2.beney.augmented_paper_detector;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentOnAttachListener;

import com.google.android.filament.Engine;
import com.google.android.filament.filamat.MaterialBuilder;
import com.google.android.filament.filamat.MaterialPackage;
import com.google.ar.core.Anchor;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.AugmentedImageDatabase;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.ImageFormat;
import com.google.ar.core.Plane;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.SceneView;
import com.google.ar.sceneform.Sceneform;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.EngineInstance;
import com.google.ar.sceneform.rendering.ExternalTexture;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInstance;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.BaseArFragment;
import com.google.ar.sceneform.ux.InstructionsController;
import com.google.ar.sceneform.ux.TransformableNode;
import com.lg2.beney.augmented_paper_detector.core.PaperEdgeDetector;
import com.lg2.beney.augmented_paper_detector.utils.ImageUtils;
import com.lg2.beney.augmented_paper_detector.utils.SnackbarHelper;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
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
        BaseArFragment.OnTapArPlaneListener,
        BaseArFragment.OnSessionConfigurationListener,
        ArFragment.OnViewCreatedListener{

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private static final boolean VDBG = false;

    private final List<CompletableFuture<Void>> futures = new ArrayList<>();
    private ArFragment arFragment;

    private Renderable model;
    private ViewRenderable viewRenderable;
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

    private final static String AR_IMAGE_DATABASE_FILENAME = "ar_image_database.imgdb";
    private PaperEdgeDetector mEdgeDetector = null;

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

        mEdgeDetector = new PaperEdgeDetector(true);

        if (USE_BACKGROUND_4_OBJECT_DETECTION) {
            mExecutorService = Executors.newFixedThreadPool(1);
        }

        loadModels();
    }

    @Override
    public void onAttachFragment(@NonNull FragmentManager fragmentManager, @NonNull Fragment fragment) {
        if (fragment.getId() == R.id.arFragment) {
            arFragment = (ArFragment) fragment;
            arFragment.setOnSessionConfigurationListener(this);
            arFragment.setOnViewCreatedListener(this);
            arFragment.setOnTapArPlaneListener(this);
            // arFragment.setOnTapPlaneGlbModel("https://storage.googleapis.com/ar-answers-in-search-models/static/Tiger/model.glb");
        }
    }

    @Override
    public void onSessionConfiguration(Session session, Config config) {

        // TODO: HitTest
        /*
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            config.setDepthMode(Config.DepthMode.AUTOMATIC);
        }

        log("getPlaneFindingMode()" + config.getPlaneFindingMode());
        */

        // TODO: HitTest
        // Disable plane detection
        // config.setPlaneFindingMode(Config.PlaneFindingMode.DISABLED);

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
    }

    @Override
    public void onViewCreated(ArSceneView arSceneView) {
        arFragment.setOnViewCreatedListener(null);

        // Fine adjust the maximum frame rate
        arSceneView.setFrameRateFactor(SceneView.FrameRate.FULL);
    }

    public void loadModels() {
        WeakReference<MainActivity> weakActivity = new WeakReference<>(this);
        ModelRenderable.builder()
                .setSource(this, Uri.parse("https://storage.googleapis.com/ar-answers-in-search-models/static/Tiger/model.glb"))
                .setIsFilamentGltf(true)
                .setAsyncLoadEnabled(true)
                .build()
                .thenAccept(model -> {
                    MainActivity activity = weakActivity.get();
                    if (activity != null) {
                        activity.model = model;
                    }
                })
                .exceptionally(throwable -> {
                    Toast.makeText(
                            this, "Unable to load model", Toast.LENGTH_LONG).show();
                    return null;
                });
        ViewRenderable.builder()
                .setView(this, R.layout.view_model_title)
                .build()
                .thenAccept(viewRenderable -> {
                    MainActivity activity = weakActivity.get();
                    if (activity != null) {
                        activity.viewRenderable = viewRenderable;
                    }
                })
                .exceptionally(throwable -> {
                    Toast.makeText(this, "Unable to load model", Toast.LENGTH_LONG).show();
                    return null;
                });
    }

    @Override
    public void onTapPlane(HitResult hitResult, Plane plane, MotionEvent motionEvent) {
        if (model == null || viewRenderable == null) {
            Toast.makeText(this, "Loading...", Toast.LENGTH_SHORT).show();
            return;
        }

        // Create the Anchor.
        Anchor anchor = hitResult.createAnchor();
        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setParent(arFragment.getArSceneView().getScene());

        // Create the transformable model and add it to the anchor.
        TransformableNode model = new TransformableNode(arFragment.getTransformationSystem());
        model.setParent(anchorNode);
        model.setRenderable(this.model)
                .animate(true).start();
        model.select();

        Node titleNode = new Node();
        titleNode.setParent(model);
        titleNode.setEnabled(false);
        titleNode.setLocalPosition(new Vector3(0.0f, 1.0f, 0.0f));
        titleNode.setRenderable(viewRenderable);
        titleNode.setEnabled(true);
    }

    private int frameIndex = -1;

    private void onUpdateFrame(FrameTime frameTime) {

        Frame frame = arFragment.getArSceneView().getArFrame();

        log(frameIndex, "onUpdateFrame: frame time(ms) = " + frameTime.getDeltaTime(TimeUnit.MICROSECONDS));

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

            /*
            // TODO: HitTest
            Mat rgbMat = ImageUtils.imageToRgbMat(image);

            final Mat debugMat = rgbMat.clone();

            final long ED_StartTime = SystemClock.uptimeMillis();
            List<MatOfPoint> convexHulls = mEdgeDetector.getConvexHulls(rgbMat, debugMat);
            List<MatOfPoint> mergedHulls = mEdgeDetector.mergeHulls(rgbMat, debugMat, convexHulls);
            final long ED_ProcessingTimeMs = SystemClock.uptimeMillis() - ED_StartTime;

            log(frameIndex, "[9toy] processImage(EDGE_DETECT): Done" +
                    ", results=" + mergedHulls.size() + ", processingTime=" + ED_ProcessingTimeMs + " ms");

            Imgproc.drawContours(rgbMat,
                    mergedHulls,
                    -1, new Scalar(255, 0, 0), 10);

            ImageUtils.SaveBitmapToFile(this, ImageUtils.imageRgbMatToBitmap(rgbMat));
            */

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

                if (!mDecode) return;

                final Mat debugMat = mImage.clone();

                final long ED_StartTime = SystemClock.uptimeMillis();
                List<MatOfPoint> convexHulls = mEdgeDetector.getConvexHulls(mImage, debugMat);
                List<MatOfPoint> mergedHulls = mEdgeDetector.mergeHulls(mImage, debugMat, convexHulls);
                final long ED_ProcessingTimeMs = SystemClock.uptimeMillis() - ED_StartTime;

                log(frameIndex, "[9toy] processImage(EDGE_DETECT): Done" +
                        ", results=" + mergedHulls.size() + ", processingTime=" + ED_ProcessingTimeMs + " ms");


                Imgproc.drawContours(mImage,
                        mergedHulls,
                        -1, new Scalar(255, 0, 0), 10);

                // ImageUtils.SaveBitmapToFile(this, ImageUtils.imageBgrMatToBitmap(mImage));

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