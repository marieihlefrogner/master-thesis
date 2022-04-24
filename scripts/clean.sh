arg=${1:-.}
exts="aux bbl blg brf idx ilg ind lof log lol lot out toc synctex.gz bcf fdb_latexmk fls pyg run.xml _minted-mymaster"

if [ -d $arg ]; then
    for ext in $exts; do
         rm -rf $arg/*.$ext
    done
else
    for ext in $exts; do
         rm -rf $arg.$ext
    done
fi