package my.capstone;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;

public class App {
    public static void main(String[] args) {
        String modelPath = "/capstone/resources/gogh2photo_G_BA.pt";
        String imagePath = "/capstone/input/summer-landscape-1330260.jpg";
        int targetWidth = 64;
        int targetHeight = 64;

        // Load the PyTorch model
        try (Model model = Model.newInstance(modelPath, Application.CV.IMAGE_CLASSIFICATION)) {

            // Create a Translator for resizing the input image
            Translator<BufferedImage, Classifications> translator = new ImageTranslator(targetWidth, targetHeight);

            // Create a Predictor for inference
            try (Predictor<BufferedImage, Classifications> predictor = model.newPredictor(translator)) {

                // Load and resize the input image
                BufferedImage image = loadImage(imagePath);
                BufferedImage resizedImage = resizeImage(image, targetWidth, targetHeight);

                // Perform inference
                Classifications result = predictor.predict(resizedImage);

                // Process the output
                for (Classification classification : result.items()) {
                    System.out.println(classification.getClassName() + ": " + classification.getProbability());
                }
            }
        } catch (IOException | ModelException | TranslateException e) {
            e.printStackTrace();
        }
    }

    private static BufferedImage loadImage(String imagePath) throws IOException {
        return ImageIO.read(Paths.get(imagePath).toFile());
    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH), 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return resizedImage;
    }

    private static class ImageTranslator implements Translator<BufferedImage, Classifications> {
        private int targetWidth;
        private int targetHeight;

        public ImageTranslator(int targetWidth, int targetHeight) {
            this.targetWidth = targetWidth;
            this.targetHeight = targetHeight;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage image) {
            // Resize the image
            BufferedImage resizedImage = resizeImage(image, targetWidth, targetHeight);

            // Convert the image to an NDArray
            NDArray array = Utils.toNDArray(ctx.getNDManager(), resizedImage, NDArray.Type.FLOAT32);
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            // Process the output
            NDArray probabilities = list.singletonOrThrow();
            return new Classifications(null, probabilities);
        }
    }
}
