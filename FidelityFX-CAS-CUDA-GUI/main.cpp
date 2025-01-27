#include "mainwindow.h"

#include <QApplication>
#include <QFile>
#include <QFontDatabase>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QFile styleFile(":/stylesheets/main.qss");
    styleFile.open(QFile::ReadOnly);
    // Apply the stylesheet
    a.setStyleSheet(QString(styleFile.readAll()));
    QFontDatabase::addApplicationFont(":/assets/fonts/TitilliumWeb-Regular.ttf");

    MainWindow w;
    w.resize(QSize(320, 160));
    w.show();
    return a.exec();
}
