python ./format_files.py

python ../../../../test-suite-sql-eval/evaluation.py --gold ../../data/outputs/gold_spider.txt --pred ../../data/outputs/pred_spider_direct.txt --db "../../../test-suite-sql-eval/database" --etype all --table ../../../test-suite-sql-eval/tables.json

python ../../../../test-suite-sql-eval/evaluation.py --gold ../../data/outputs/gold_spider.txt --pred ../../data/outputs/pred_spider_amr.txt --db "../../../test-suite-sql-eval/database" --etype all --table ../../../test-suite-sql-eval/tables.json --amr

python ./format_results.py