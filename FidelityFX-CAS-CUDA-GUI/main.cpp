#include "mainwindow.h"

#include <QApplication>
#include <QFile>
#include <QFontDatabase>
#include <QIcon>
#include <QSize>
#include <QString>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setWindowIcon(QIcon(":/assets/gpuopen.ico"));

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
