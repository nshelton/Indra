// Minimal entry that boots the app and renders the main screen
#include "app/App.h"
#include "screens/MainScreen.h"

int main() {
    App app(2200, 1600, "Indra");
    MainScreen screen;
    app.run(screen);
    return 0;
}


