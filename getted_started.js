console.log("Hello Tensorflow");

const tf = require("@tensorflow/tfjs");

const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([4, 5, 6]);

a.dot(b).print();