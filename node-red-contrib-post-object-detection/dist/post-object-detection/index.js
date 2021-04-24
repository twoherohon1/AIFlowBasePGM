"use strict";
var tf = require("@tensorflow/tfjs-node");
var https_1 = require("https");
var http_1 = require("http");
function request(url, callback) {
    var reqOpt = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method: 'GET'
    };
    if (url.protocol === 'http:') {
        return http_1.request(reqOpt, callback);
    }
    else if (url.protocol === 'https:') {
        return https_1.request(reqOpt, callback);
    }
    return undefined;
}
function fetchClasses(urlStr) {
    var url;
    try {
        url = new URL(urlStr);
    }
    catch (e) {
        return Promise.reject('Invalid URL');
    }
    if (url.protocol === 'file:') {
        return Promise.resolve(require(url.pathname));
    }
    return new Promise(function (resolve, reject) {
        var chunks = [];
        var req = request(url, function (res) {
            res.setEncoding('utf8');
            res.on('data', function (chunk) { return chunks.push(chunk); });
            res.on('end', function () {
                if (res.statusCode === 200) {
                    resolve(JSON.parse(chunks.join('')));
                }
                else {
                    reject("can not fetch classes file: " + res.statusCode + ", " + res.statusMessage);
                }
            });
        });
        if (req === undefined) {
            reject('unsupported protocol');
        }
        else {
            req.on('error', function (e) {
                reject(e.message);
            });
            req.end();
        }
    });
}
module.exports = function init(RED) {
    var PostObjectDetection = (function () {
        function PostObjectDetection(config) {
            var _this = this;
            this.maxNumBoxes = 20;
            this.classesURL = config.classesURL.trim();
            config.iou = config.iou || '0.5';
            config.minScore = config.minScore || '0.5';
            this.iou = parseFloat(config.iou);
            this.minScore = parseFloat(config.minScore);
            RED.nodes.createNode(this, config);
            this.on('input', function (msg) {
                var data = msg.payload;
                _this.handleRequest(data, msg);
            });
            this.on('close', function (done) {
                _this.handleClose(done);
            });
            this.getClassName = function (idx) {
                return "" + idx;
            };
            if (this.classesURL.length > 0) {
                fetchClasses(this.classesURL)
                    .then(function (rev) { return _this.classes = rev; })
                    .then(function () {
                    _this.status({});
                    _this.getClassName = function (idx) {
                        return _this.classes[idx];
                    };
                })
                    .catch(function (error) {
                    _this.log('can not fetch classes file, using id as class name instead');
                    _this.status({ fill: 'red', shape: 'dot', text: error });
                });
            }
        }
        PostObjectDetection.prototype.handleRequest = function (inputs, origMsg) {
            var _this = this;
            var scores = inputs[0].dataSync();
            var boxes = inputs[1].dataSync();
            tf.dispose(inputs);
            var _a = this.calculateMaxScores(scores, inputs[0].shape[1], inputs[0].shape[2]), maxScores = _a[0], classes = _a[1];
            var indexTensor = tf.tidy(function () {
                var boxes2 = tf.tensor2d(boxes, [inputs[1].shape[1], inputs[1].shape[3]]);
                return tf.image.nonMaxSuppression(boxes2, maxScores, _this.maxNumBoxes, _this.iou, _this.minScore);
            });
            var indexes = indexTensor.dataSync();
            indexTensor.dispose();
            var obj = this.buildDetectedObjects(boxes, maxScores, indexes, classes);
            origMsg.payload = obj;
            this.send(origMsg);
        };
        PostObjectDetection.prototype.handleClose = function (done) {
            done();
        };
        PostObjectDetection.prototype.buildDetectedObjects = function (boxes, scores, indexes, classes) {
            var count = indexes.length;
            var objects = [];
            for (var i = 0; i < count; i++) {
                var bbox = [
                    boxes[indexes[i] * 4],
                    boxes[indexes[i] * 4 + 1],
                    boxes[indexes[i] * 4 + 2],
                    boxes[indexes[i] * 4 + 3]
                ];
                objects.push({
                    bbox: [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    className: this.getClassName(classes[indexes[i]]),
                    score: scores[indexes[i]]
                });
            }
            return objects;
        };
        PostObjectDetection.prototype.calculateMaxScores = function (scores, numBoxes, numClasses) {
            var maxes = [];
            var classes = [];
            for (var i = 0; i < numBoxes; i++) {
                var max = Number.MIN_VALUE;
                var index = -1;
                for (var j = 0; j < numClasses; j++) {
                    if (scores[i * numClasses + j] > max) {
                        max = scores[i * numClasses + j];
                        index = j;
                    }
                }
                maxes[i] = max;
                classes[i] = index;
            }
            return [maxes, classes];
        };
        return PostObjectDetection;
    }());
    RED.nodes.registerType('post-object-detection', PostObjectDetection);
};
//# sourceMappingURL=index.js.map