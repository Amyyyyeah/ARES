for i in {0..3}
do
    python -u ./preprocessing_after_RL/processing_sentence_level_feedback.py --k_actions $i --problems scienceqa
done

