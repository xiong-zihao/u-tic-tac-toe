class TreeNode:
    def __init__(self, parent=None, children=None):
        self.children = children
        self.parent = parent

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children is None

    def add_child(self, node):
        if self.children is None:
            self.children = []
        self.children.append(node)
        node.parent = self
