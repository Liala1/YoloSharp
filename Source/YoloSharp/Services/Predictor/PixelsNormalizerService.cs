namespace Compunet.YoloSharp.Services;

internal class PixelsNormalizerService : IPixelsNormalizerService
{
    public void NormalizerPixelsToTensor(Image<Rgb24> image, MemoryTensor<float> tensor, Vector<int> padding)
    {
        // Verify tensor dimensions
        if (image.Height + (padding.Y * 2) != tensor.Dimensions[2] && image.Width + (padding.X * 2) != tensor.Dimensions[3])
        {
            throw new InvalidOperationException("The image size and target tensor dimensions is not match");
        }

        // Process core
        ProcessToTensorCore(image, tensor, padding);
    }

    private static void ProcessToTensorCore(Image<Rgb24> image, MemoryTensor<float> tensor, Vector<int> padding)
    {
        var width = image.Width;
        var height = image.Height;

        // Pre-calculate strides for performance
        var strideY = tensor.Strides[2];
        var strideX = tensor.Strides[3];
        var strideR = tensor.Strides[1] * 0;
        var strideG = tensor.Strides[1] * 1;
        var strideB = tensor.Strides[1] * 2;

        Parallel.For(0, image.Height, y =>
        {
            var memory = image.DangerousGetPixelRowMemory(y);
            var pixels = memory.Span;
            var tensorSpan = tensor.Buffer.Span;
            
            for (var index = 0; index < width; ++index)
            {
                var x = index % width;

                var tensorIndex = strideR + strideY * (y + padding.Y) + strideX * (x + padding.X);

                var pixel = pixels[index];

                WritePixel(tensorSpan, tensorIndex, pixel, strideR, strideG, strideB);
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WritePixel(Span<float> target, int index, Rgb24 pixel, int strideBatchR, int strideBatchG, int strideBatchB)
    {
        target[index] = pixel.R / 255f;
        target[index + strideBatchG - strideBatchR] = pixel.G / 255f;
        target[index + strideBatchB - strideBatchR] = pixel.B / 255f;
    }
}