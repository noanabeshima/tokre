import lark
from lark import Lark, Transformer, Tree, Discard
import regex as re
from modules import Toks, Repeat, Phrase, OrGroup, VarDefn, VarRef, Wildcard
import tiny_model
from .macros import DEFINED_MACROS

grammar = r"""
    %import common.INT

    SPACE: " "
    SPACES: SPACE+
    pad: SPACES

    NUMBER: INT
    WILDCARD: /[.]/
    ESCAPED: /\\[*+|.()\[\]{}?=<>\\0-9]/

    REPEAT_POSTFIX.1: /[+*?]/ | "{" SPACES? ( NUMBER? SPACES? "," SPACES? NUMBER | NUMBER SPACES? "," NUMBER? | NUMBER) SPACES? "}"

    ?non_special_chars: /[^*+|.()\[\]{}?=<>\\0-9]/
    char_seq: (non_special_chars | ESCAPED | NUMBER | SPACES)+

    pos_lookbehind: "(?<=" pattern ")"
    pos_lookahead: "(?=" pattern ")"
    neg_lookbehind: "(?<!" pattern ")"
    neg_lookahead: "(?!" pattern ")"

    ?lookaround: (pos_lookbehind | pos_lookahead | neg_lookbehind | neg_lookahead)
    
    identifier_chars: /[a-zA-Z_]/
    identifier: identifier_chars+

    macro_string_arg: "`" (/[^`]/)* "`"
    bool: /False|True/
    ?macro_arg: (group_block | macro_string_arg | NUMBER | bool)
    macro_kwarg: identifier "=" macro_arg
    macro: "[" pad? identifier (pad (macro_arg| macro_kwarg))* pad? "]"

    var_defn: "[" pad? identifier pad? "=" pattern "]"

    #  <group_block>{..}
    repeated_group: group_block REPEAT_POSTFIX

    # clearly delineated block
    # (...) | (..|..) | [...] | . | [..=..] | ..among others..
    ?group_block: "(" (or_pattern | phrase) ")" | macro | var_defn | lookaround | repeated_group | WILDCARD
    
    or_pattern: phrase ("|" phrase)+

    # ( group | group{...} | seq of chars)+
    ?phrase: (group_block  | char_seq )+

    ?pattern: phrase | or_pattern

    NEWLINE: "\n"

    # line_defn: identifier pad? "=" " (Lily)"
    
    ?lines: (NEWLINE* (pattern)) (NEWLINE+ (pattern))* NEWLINE*

    ?start   : lines
"""

def escape(s):
    """
    Escapes special characters in a string for use in tokre patterns.
    
    Args:
    s (str): The input string to escape.
    
    Returns:
    str: The escaped string.
    """
    special_chars_wo_newline = r"\^$|&*+{}()<>?[].=;" + ''.join([str(i) for i in range(10)])
    s = ''.join('\\' + c if c in special_chars_wo_newline else c for c in s)
    s = s.replace("\n", "\\n")
    return s


parser = Lark(grammar, parser="lalr")


class SimplifyTree(Transformer):
    def __init__(self):
        super().__init__()
        self.line_defns = {}
        self.var_defns = {}

    def pad(self, items):
        return Discard

    def NEWLINE(self, items):
        return Discard

    def repeated_group(self, items):
        assert len(items) == 2, items
        group, repeat_str = items[0], items[1]

        if repeat_str == "+":
            min, max = 1, float("inf")
        elif repeat_str == "*":
            min, max = 0, float("inf")
        elif repeat_str == "?":
            min, max = 0, 1
        else:
            assert repeat_str[0] == r"{" and repeat_str[-1] == r"}"

            # remove curly brackets
            repeat_str = repeat_str[1:-1]

            if "," in repeat_str:
                min, max = repeat_str.split(",")
                min, max = min.strip(), max.strip()
                min = 0 if min == "" else int(min)
                max = float("inf") if max == "" else int(max)
            else:
                num = int(repeat_str.strip())
                min, max = num, num

        assert not (min > max), f"min > max in parsed repeat: {min=} {max=}"

        # return Repeat(child_matcher=group, min=min, max=max)
        return Tree("repeat", [group, min, max])

    def char_seq(self, children):
        s = ""
        for child in children:
            assert isinstance(child, lark.lexer.Token)
            child_str = str(child)

            if len(child_str) == 2 and child_str[0] == "\\":
                child_str = child_str[-1]

            s += child_str
        if len(s.strip()) == 0:
            return Discard

        s = s.rstrip()
        if s[0] == " ":
            s = " " + s.lstrip()
        s.replace(r"\n", "\n")
        return Tree("string", [s])

    def identifier_chars(self, chars):
        s = "".join([str(c) for c in chars])
        return s

    def identifier(self, chars):
        return "".join(chars)

    def macro(self, args):
        args = [
            arg
            for arg in args
            if not (isinstance(arg, lark.lexer.Token) and arg.type == "SPACES")
        ]
        macro_name = args[0]
        assert isinstance(macro_name, str)

        if macro_name in self.var_defns:
            return Tree("var_ref", [macro_name])
        elif macro_name in self.line_defns:
            return self.line_defns[macro_name]
        else:
            return Tree("macro", [macro_name, *args[1:]])

    def pattern(self, args):
        return Tree("pat", args)

    def var_defn(self, children):
        children = [
            child
            for child in children
            if not (isinstance(child, lark.lexer.Token) and child.type == "SPACES")
        ]
        assert len(children) == 2
        var_name, child_tree = children

        assert isinstance(var_name, str)
        self.var_defns[var_name] = child_tree

        child_tree = children[1]

        return Tree("var_defn", [var_name, child_tree])

    def pos_lookahead(self, children):
        assert len(children) == 1
        child_tree = children[0]

        # [child, is_backward, is_neg]
        return Tree("lookaround", [child_tree, False, False])

    def pos_lookbehind(self, children):
        assert len(children) == 1
        child_tree = children[0]

        # [child, is_backward, is_neg]
        return Tree("lookaround", [child_tree, True, False])

    def neg_lookahead(self, children):
        assert len(children) == 1
        child_tree = children[0]

        # [child, is_backward, is_neg]
        return Tree("lookaround", [child_tree, False, True])

    def neg_lookbehind(self, children):
        assert len(children) == 1
        child_tree = children[0]

        # [child, is_backward, is_neg]
        return Tree("lookaround", [child_tree, False, False])

    def macro_string_arg(self, children):
        return "".join([str(ch) for ch in children])

    def or_pattern(self, children):
        return Tree("or_pattern", children)

    def line_defn(self, children):
        assert len(children) == 2
        defn_name, child_tree = children
        self.line_defns[defn_name] = child_tree
        return Discard

    def lines(self, lines):
        if len(lines) > 1:
            return Tree("phrase", lines)
        else:
            assert (
                len(lines) == 1
            ), "Don't see at least at least one line of code; did you feed in only line definitions?"
            return lines[0]

    def phrase(self, children):
        return Tree("phrase", children)

    def bool(self, children):
        assert len(children) == 1
        val = bool(str(children[0]))
        return val

    def NUMBER(self, num_token):

        val = int(str(num_token))
        return val

    def macro_kwarg(self, children):
        assert len(children) == 2
        assert isinstance(children[0], str)

        return {children[0]: children[1]}

    def WILDCARD(self, chilren):
        """
        need to convert to tree before InsertModules
        or an error gets thrown if a tokre-code line is only a wildcard.
        Can't apply a Transformer to a Token object.
        """
        return Tree("wildcard", [])


class InsertModules(Transformer):
    def repeat(self, children):
        assert len(children) == 3
        child_matcher, repeat_min, repeat_max = children
        return Repeat(child_matcher=child_matcher, min=repeat_min, max=repeat_max)

    def string(self, children):
        assert len(children) == 1, children
        assert isinstance(children[0], str)
        toks = tiny_model.raw_toks[
            tiny_model.enc(stories=children)[0].tolist()
        ].tolist()
        return Toks(toks=toks)

    def wildcard(self, children):
        return Wildcard()

    def phrase(self, children):
        return Phrase(matchers=children)

    def or_pattern(self, children):
        return OrGroup(matchers=children)

    def var_defn(self, children):
        assert len(children) == 2
        var_name, child_matcher = children
        return VarDefn(var_name=var_name, child_matcher=child_matcher)

    def var_ref(self, children):
        assert len(children) == 1, children
        var_name = children[0]
        return VarRef(var_name=var_name)

    def macro(self, children):
        assert len(children) >= 1

        macro_name, children = children[0], children[1:]

        args, kwargs = [], {}

        for child in children:
            if isinstance(child, dict):
                kwargs = kwargs | child
            else:
                args.append(child)

        if macro_name in DEFINED_MACROS:
            return DEFINED_MACROS[macro_name](*args, **kwargs)
        else:
            assert False, f"macro {macro_name} not found in macros.py"


def parse_line(line: str):
    assert "\n" not in line
    # [non-newline space]* [name=...] [non-newline space]*
    match = re.match(r"[^\S\r\n]*(?P<name>[a-zA-Z_]+)[^\S\r\n]*=", line)
    if match:
        assert "name" in match.groupdict()
        defn_name = match.groupdict()["name"]
        pattern = line[match.end() :]
        return Tree("line_defn", [defn_name, parser.parse(pattern)])
    else:
        return parser.parse(line)


def parse(s, return_tree=False):
    parsed_lines = [parse_line(line) for line in s.split("\n") if len(line.strip()) > 0]
    tree = Tree("lines", parsed_lines)
    tree = SimplifyTree().transform(tree)
    if return_tree:
        return tree
    matcher = InsertModules().transform(tree)
    return matcher
