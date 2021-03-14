console.log("Hello Tensorflow");

const tf = require("@tensorflow/tfjs");

function onEpochEnd(data, logs) {
    console.log("loss", logs.loss);
}

async function main() {
    const N = 10000;
    const data   = Array.from(new Array(N).keys());
    const answer = data.map(x => x*2);

    const xs = tf.tensor2d(data, [N, 1]);
    const ys = tf.tensor2d(answer, [N, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [1],
        units: 1,
        activation: "relu",
        kernelInitializer: "ones"
    }));
    model.compile({
        optimizer: tf.train.adam(0.5),
        loss: 'meanSquaredError'
    });

    const h = await model.fit(xs, ys, {
       epochs: 100,
       batchSize: 100,
        callbacks: {
           onEpochEnd
        }
    });

    model.predict(tf.tensor2d([[10001], [10002]])).print();
//    model.predict(tf.tensor2d([[-1]])).print();
}

main();