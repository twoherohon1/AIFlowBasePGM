declare type NodeRed = {
    nodes: NodeRedNodes;
};
declare type NodeRedNodes = {
    createNode(node: any, props: NodeRedProperties): void;
    registerType(type: string, ctor: any): void;
};
declare type NodeRedProperties = {
    url: string;
};
declare const _default: (RED: NodeRed) => void;
export = _default;
