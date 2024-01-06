function feats = cnn(this_rgb)
    net = resnet50;
    layer = 'avg_pool';
    input_size = net.Layers(1).InputSize;

    % Asegúrate de que this_rgb sea una imagen válida de MATLAB
    % Puedes necesitar convertirla desde el formato cv::Mat a un formato de MATLAB
    im_scale = imresize(this_rgb, [input_size(1), input_size(2)]);

    feats_spt_deep = activations(net, im_scale, layer, 'ExecutionEnvironment','cpu');

    % Aplana el resultado para que sea un vector 1D
    feats = squeeze(feats_spt_deep);
end


// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>

// cv::Mat cnn(const cv::Mat& this_rgb) {
//     // Carga el modelo ResNet-50 preentrenado
//     cv::dnn::Net net = cv::dnn::readNetFromONNX("resnet50.onnx");

//     // Prepara la imagen de entrada
//     cv::Mat blob;
//     cv::Size inputSize(224, 224); // Tamaño típico para ResNet-50
//     cv::dnn::blobFromImage(this_rgb, blob, 1.0, inputSize, cv::Scalar(104, 117, 123));

//     // Pasada hacia adelante
//     net.setInput(blob);
//     cv::Mat result = net.forward("avg_pool");

//     // Aplana el resultado a un vector 1D
//     cv::Mat feats = result.reshape(1, 1);

//     return feats;
// }