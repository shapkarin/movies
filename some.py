array = 'sone<EOS>other<EOS>woooooo'.split('<EOS>')
plots = open('data/plots', 'r')

plots_array = plots.read().split('<EOS>')
titles_array = list(open('data/titles', 'r'))


result = open('result', 'w')
result.write(titles_array[2])
result.write(plots_array[2])