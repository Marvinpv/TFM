import note_seq
import json

PATH_INFERENCE = '/home/marvin/US/TFM/code/infer/tmp-jazz_solos_notes_ties_vb1_test-00000-of-00001/jazz_solos_notes_ties_vb1_test-predict.jsonl-00000-of-00001-chunk00000'

if __name__ == '__main__':
    with open(PATH_INFERENCE) as f:
        inference = json.loads(f.read())
    #print('Inference:',inference)
    note_sequence = note_seq.NoteSequence()

    # Iterar sobre la lista de diccionarios y agregar cada nota al NoteSequence
    for note_dict in inference['est_notes']:
        note = note_sequence.notes.add()
        note.start_time = note_dict["start_time"]
        note.end_time = note_dict["end_time"]
        note.pitch = note_dict["pitch"]
        note.velocity = note_dict["velocity"]
        note.program = note_dict["program"]
        note.is_drum = note_dict["is_drum"] 

    # (Opcional) Establecer otras propiedades del NoteSequence, como tempo, tiempo de creación, etc.
    note_sequence.tempos.add(qpm=120)  # Establecer el tempo (QPM: quarter notes per minute)
    note_sequence.total_time = max(note.end_time for note in note_sequence.notes)  # Calcular el tiempo total

    note_seq.play_sequence(note_sequence)
