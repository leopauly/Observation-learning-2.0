#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -l coproc_k80=2
#$ -M cnlp@leeds.ac.uk
#$ -m be

echo "Starting audio classifier training script..."

module load python
module load python-libs/3.1.0

python < 3.training_cnn_melspec.py >> 3.training_cnn_melspec.txt