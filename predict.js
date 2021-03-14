console.log("Hello Tensorflow");

const tf = require("@tensorflow/tfjs");
const tfnode = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

async function main(file) {
    const model = await tf.loadLayersModel("file:///cnn-model10/model.json");
    let buffer = fs.readFileSync(file);
    let tfimage = tfnode.node.decodeImage(buffer, chanels=3);
    tfimage = tf.image.resizeBilinear(tfimage, [28, 28]);
    tfimage = tfimage.cast("float32").div(255);

    const pred = model.predict(tf.stack([tfimage]));
    pred.print();
}

main("./data/seg_test/seg_test/sea/20076.jpg");
main("./data/seg_test/seg_test/sea/20106.jpg");
main("./data/seg_test/seg_test/forest/20056.jpg");
main("./data/seg_test/seg_test/forest/20091.jpg");

