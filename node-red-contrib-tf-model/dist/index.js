"use strict";
var os_1 = require("os");
var path_1 = require("path");
var fs_1 = require("fs");
var tf = require("@tensorflow/tfjs-node");
var node_fetch_1 = require("node-fetch");
var url_1 = require("url");
var CACHE_DIR = path_1.join(os_1.homedir(), '.node-red', 'tf-model');
fs_1.mkdirSync(CACHE_DIR, { recursive: true });
var MODEL_CACHE_ENTRIES = path_1.join(CACHE_DIR, 'models.json');
var gModelCache = fs_1.existsSync(MODEL_CACHE_ENTRIES) ?
    require(MODEL_CACHE_ENTRIES) : {};
if (Object.getOwnPropertyNames(gModelCache).length === 0) {
    updateCacheEntries(MODEL_CACHE_ENTRIES);
}
function updateCacheEntries(filename) {
    fs_1.writeFileSync(filename, JSON.stringify(gModelCache, null, 2));
}
function hashCode(str) {
    var hash = 0, i, chr;
    if (str === undefined || str.length === 0) {
        return "" + hash;
    }
    for (i = 0; i < str.length; i++) {
        chr = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr | 0;
    }
    return "" + hash;
}
function fetchAndStore(url, filePath) {
    return new Promise(function (resolve, reject) {
        node_fetch_1.default(url).then(function (res) { return res.buffer(); })
            .then(function (buff) {
            fs_1.writeFile(filePath, buff, function (err) {
                if (err) {
                    reject(err);
                }
                else {
                    resolve(filePath);
                }
            });
        });
    });
}
function fetchNewModelFiles(url) {
    var filename = 'model.json';
    var modelFile;
    var hash = hashCode(url);
    var modelFolder = path_1.join(CACHE_DIR, hash);
    var newCache = { hash: hash, filename: filename, lastModified: '' };
    return node_fetch_1.default(url)
        .then(function (res) {
        newCache.lastModified = res.headers.get('last-modified');
        return res.buffer();
    })
        .then(function (body) {
        return new Promise(function (resolve, reject) {
            fs_1.mkdirSync(modelFolder, { recursive: true });
            modelFile = path_1.join(modelFolder, filename);
            fs_1.writeFile(modelFile, body, function (err) {
                if (err) {
                    reject(err);
                }
                else {
                    gModelCache[url] = newCache;
                    updateCacheEntries(MODEL_CACHE_ENTRIES);
                    resolve(require(modelFile));
                }
            });
        });
    })
        .then(function (model) {
        if (model.weightsManifest !== undefined) {
            var parsedURL_1 = url_1.parse(url);
            var dir_1 = path_1.dirname(parsedURL_1.pathname);
            var allFetch_1 = [];
            model.weightsManifest[0].paths.forEach(function (shardFile) {
                parsedURL_1.pathname = dir_1 + "/" + shardFile;
                allFetch_1.push(fetchAndStore(parsedURL_1.protocol + "//" + parsedURL_1.host + parsedURL_1.pathname, path_1.join(modelFolder, shardFile)));
            });
            return Promise.all(allFetch_1);
        }
        return Promise.resolve([]);
    })
        .then(function () { return modelFile; });
}
function removeCacheEntry(urlStr) {
    var entry = gModelCache[urlStr];
    if (entry !== undefined) {
        var modelFolder_1 = path_1.join(CACHE_DIR, entry.hash);
        var files = fs_1.readdirSync(modelFolder_1);
        files.forEach(function (file) {
            fs_1.unlinkSync(path_1.join(modelFolder_1, file));
        });
        fs_1.rmdirSync(modelFolder_1);
        delete gModelCache[urlStr];
        updateCacheEntries(MODEL_CACHE_ENTRIES);
    }
}
function downloadOrUpdateModelFiles(urlStr, cacheFirst) {
    if (cacheFirst === void 0) { cacheFirst = true; }
    var url;
    try {
        url = new URL(urlStr);
    }
    catch (e) {
        return Promise.reject('Invalid URL');
    }
    if (url.protocol === 'file:') {
        return Promise.resolve(url_1.fileURLToPath(urlStr));
    }
    var cacheEntry = gModelCache[urlStr];
    if (cacheEntry !== undefined) {
        return node_fetch_1.default(urlStr, {
            headers: { 'If-Modified-Since': cacheEntry.lastModified },
            method: 'HEAD',
        })
            .then(function (res) {
            if (res.status === 304) {
                return path_1.join(CACHE_DIR, cacheEntry.hash, cacheEntry.filename);
            }
            if (res.status === 200) {
                return fetchNewModelFiles(urlStr);
            }
            else {
                if (cacheFirst) {
                    return path_1.join(CACHE_DIR, cacheEntry.hash, cacheEntry.filename);
                }
                throw new Error("can not retrieve model: " + res.statusText);
            }
        })
            .catch(function (e) {
            throw e;
        });
    }
    else {
        return fetchNewModelFiles(urlStr);
    }
}
module.exports = function tfModel(RED) {
    var TFModel = (function () {
        function TFModel(config) {
            var _this = this;
            this.id = config.id;
            this.type = config.type;
            this.name = config.name;
            this.wires = config.wires;
            this.modelURL = config.modelURL;
            this.outputNode = config.outputNode || '';
            RED.nodes.createNode(this, config);
            this.on('input', function (msg) {
                _this.handleRequest(msg.payload, msg);
            });
            this.on('close', function (done) {
                _this.handleClose(done);
            });
            if (this.modelURL.trim().length > 0) {
                downloadOrUpdateModelFiles(this.modelURL)
                    .then(function (modelPath) {
                    _this.status({ fill: 'red', shape: 'ring', text: 'loading model...' });
                    _this.log("loading model from: " + _this.modelURL);
                    var modelJson = require(modelPath);
                    var rev;
                    if (modelJson.format === 'layers') {
                        rev = tf.loadLayersModel(tf.io.fileSystem(modelPath));
                    }
                    else {
                        rev = tf.loadGraphModel(tf.io.fileSystem(modelPath));
                    }
                    return rev;
                })
                    .then(function (model) {
                    _this.model = model;
                    _this.status({
                        fill: 'green',
                        shape: 'dot',
                        text: 'model is ready'
                    });
                    _this.log("model loaded");
                    if (_this.model instanceof tf.GraphModel) {
                        _this.log("input(s) for the model: " + JSON.stringify(_this.model.inputNodes));
                    }
                    else {
                        _this.log("input(s) for the model: " + JSON.stringify(_this.model.inputNames));
                    }
                })
                    .catch(function (e) {
                    _this.error(e.message);
                    _this.status({
                        fill: 'red',
                        shape: 'dot',
                        text: "failed to load the model: " + e.message
                    });
                    _this.handleError(e);
                });
            }
        }
        TFModel.prototype.handleRequest = function (inputs, origMsg) {
            var _this = this;
            if (!this.model) {
                this.error("model is not ready");
                return;
            }
            var result;
            if (this.model instanceof tf.GraphModel) {
                result = this.model.executeAsync(inputs, this.outputNode);
            }
            else {
                result = Promise.resolve(this.model.predict(inputs));
            }
            result.then(function (result) {
                var msg = origMsg;
                msg.payload = result;
                _this.send(msg);
                _this.cleanUp(inputs);
            })
                .catch(function (e) {
                _this.error(e.message);
                _this.cleanUp(inputs);
            });
        };
        TFModel.prototype.handleError = function (error) {
            var msg = error.message || '';
            if (msg.indexOf('byte length of Float32Array should be a multiple of 4') !== -1) {
                removeCacheEntry(this.modelURL);
                this.error('Model files are corrupted, restart this node to redownload the model again');
            }
        };
        TFModel.prototype.cleanUp = function (tensors) {
            tf.dispose(tensors);
        };
        TFModel.prototype.handleClose = function (done) {
            if (this.model) {
                this.model.dispose();
            }
            done();
        };
        return TFModel;
    }());
    RED.nodes.registerType('tf-model', TFModel);
};
//# sourceMappingURL=index.js.map