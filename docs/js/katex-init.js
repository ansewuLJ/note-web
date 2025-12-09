document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: "$", right: "$", display: false},   // 行内公式 $ ... $
            {left: "$$", right: "$$", display: true},   // 块级公式 $$ ... $$
            {left: "\\(", right: "\\)", display: false}, // 行内公式 \( ... \)
            {left: "\\[", right: "\\]", display: true}   // 块级公式 \[ ... \]
        ],
        throwOnError: false
    });
});
