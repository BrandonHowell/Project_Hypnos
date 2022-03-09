import PySimpleGUI as sg

input('Some INput')

layout = [[sg.Text("What's your name?")],
          [sg.Input(key='-INPUT-')],
          [sg.Text(size=(40,1), key='-OUTPUT-')],
          [sg.Button('Ok'), sg.Button('Quit')]]

window = sg.Window('Window Title', layout)

while True:
    event, values = window.read()
    
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "!")
    
window.close()