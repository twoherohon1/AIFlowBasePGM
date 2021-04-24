"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var bert_tokenizer_1 = require("bert-tokenizer");
var node_fetch_1 = __importDefault(require("node-fetch"));
var fs_1 = __importDefault(require("fs"));
var path_1 = __importDefault(require("path"));
var url_1 = __importDefault(require("url"));
module.exports = function (RED) {
    var BertTokensNode = (function () {
        function BertTokensNode(config) {
            var _this = this;
            this._localPath = "node_modules/node-red-contrib-bert-tokenizer/node_modules/bert-tokenizer/assets/vocab.json";
            this.loadVocabulary(config.url);
            RED.nodes.createNode(this, config);
            this.on('input', function (msg) {
                _this.handleRequest(msg);
            });
        }
        ;
        BertTokensNode.prototype.fetchVocab = function (url, dir) {
            return __awaiter(this, void 0, void 0, function () {
                var target, res, fileStream;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            target = path_1.default.join(dir, 'vocab.json');
                            return [4, node_fetch_1.default(url)];
                        case 1:
                            res = _a.sent();
                            fileStream = fs_1.default.createWriteStream(target);
                            return [2, new Promise(function (resolve, reject) {
                                    res.body.pipe(fileStream);
                                    res.body.on("error", function (err) {
                                        reject(err);
                                    });
                                    fileStream.on("finish", function () {
                                        resolve();
                                    });
                                })];
                    }
                });
            });
        };
        BertTokensNode.prototype.loadVocabulary = function (vocabUrl) {
            return __awaiter(this, void 0, void 0, function () {
                var url, err_1;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!(vocabUrl != "")) return [3, 6];
                            url = url_1.default.parse(vocabUrl, false, true);
                            _a.label = 1;
                        case 1:
                            _a.trys.push([1, 5, , 6]);
                            if (!(url.protocol === "http:" || url.protocol === "https:")) return [3, 3];
                            return [4, this.fetchVocab(vocabUrl, __dirname)];
                        case 2:
                            _a.sent();
                            this.localPath = path_1.default.join(__dirname, 'vocab.json');
                            console.log("Downloaded custom vocabulary file from " + vocabUrl + ".");
                            return [3, 4];
                        case 3:
                            if (url.protocol === "file:") {
                                if (fs_1.default.existsSync(url_1.default.fileURLToPath(vocabUrl))) {
                                    this.localPath = url.path;
                                    console.log("Found vocabulary " + url.path + ".");
                                }
                                else {
                                    throw url.path + " file does not exist. Try use a absolute path.";
                                }
                            }
                            else {
                                throw "Unsupported url format " + vocabUrl + ". The url should start with file://, http:// or, https://.";
                            }
                            _a.label = 4;
                        case 4: return [3, 6];
                        case 5:
                            err_1 = _a.sent();
                            this.error(err_1 + " Using default vocabulary " + this.localPath + " instead.");
                            return [3, 6];
                        case 6:
                            this.bertTokenizer = (vocabUrl === "") ? new bert_tokenizer_1.BertTokenizer() : new bert_tokenizer_1.BertTokenizer(this.localPath, true);
                            console.log("Vocabulary Loaded.");
                            return [2];
                    }
                });
            });
        };
        Object.defineProperty(BertTokensNode.prototype, "localPath", {
            get: function () {
                return this._localPath;
            },
            set: function (url) {
                this._localPath = url;
            },
            enumerable: true,
            configurable: true
        });
        BertTokensNode.prototype.handleRequest = function (inMsg) {
            var outMsg = Object.assign({}, inMsg);
            outMsg.payload = this.bertTokenizer.convertSingleExample(inMsg.payload);
            console.log(outMsg.payload);
            this.send(outMsg);
        };
        return BertTokensNode;
    }());
    RED.nodes.registerType("bert-tokenizer", BertTokensNode);
};
//# sourceMappingURL=index.js.map